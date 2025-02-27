# Import necessary libraries
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.metrics import root_mean_squared_error
from torch.optim.swa_utils import AveragedModel, SWALR

# Set random seed for reproducibility
np.random.seed(42)
torch.manual_seed(42)


# 1.Load Historical Data for Later Use
def load_historical_data(ticker, start_date, end_date, num_steps):
    # Fetch data from Yahoo Finance
    df = yf.download(ticker, start=start_date, end=end_date)
    prices = df['Close'].values
    # Split into paths of length num_steps
    num_paths = len(prices) // num_steps
    paths = [prices[i * num_steps:(i + 1) * num_steps] for i in range(num_paths)]
    return torch.tensor(np.array(paths), dtype=torch.float32)


# Load historical data
historical_data = load_historical_data('AAPL', '2020-01-01', '2022-01-01', 101)


# 2. Data Generation - Monte Carlo Simulations
def generate_synthetic_data(num_paths=1000, num_steps=100, dt=0.01, mu=0.05, sigma=0.2):
    S0 = 100  # Initial asset price
    paths = np.zeros((num_paths, num_steps + 1))
    paths[:, 0] = S0
    for t in range(1, num_steps + 1):
        Z = np.random.normal(0, 1, num_paths)
        paths[:, t] = paths[:, t - 1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
    return torch.tensor(paths, dtype=torch.float32)


# Generate synthetic data
num_paths = 1000
num_steps = 100
data = generate_synthetic_data(num_paths=num_paths, num_steps=num_steps)

# Normalize data
data_mean = data.mean(dim=1, keepdim=True)
data_std = data.std(dim=1, keepdim=True)
data = (data - data_mean) / data_std


# 3. Reinforcement Learning Agent for Optimal Measure Change
class PolicyNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(torch.relu(self.fc2(x)))
        return torch.clamp(self.fc3(x), min=-0.25, max=0.25)  # Reduced clipping range for drift adjustment


# Define the policy network
input_dim = num_steps  # Each path has num_steps features
output_dim = num_steps  # Output is the adjusted drift for each time step
policy_net = PolicyNetwork(input_dim, output_dim)
optimizer = optim.Adam(policy_net.parameters(), lr=0.005, weight_decay=5e-6)  # Reduce weight decay
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10,
                                                       factor=0.8)  # Less aggressive scheduler

# Define SWA components
swa_model = AveragedModel(policy_net)
swa_scheduler = SWALR(optimizer, swa_lr=0.005)
swa_start = 75  # Start SWA after 75 epochs


# 4. Loss Function and Training Loop for Girsanov's Theorem Optimization
def policy_gradient_loss(log_probs, rewards):
    return -torch.mean(log_probs * rewards)  # Averaging over log_probs and rewards for stability


# Modified reward function
def modified_reward_function(path, predicted_path):
    # Calculate returns based on predicted path
    returns = (predicted_path[1:] - predicted_path[:-1]) / predicted_path[:-1]
    mean_return = torch.mean(returns)
    std_return = torch.std(returns)

    # Small epsilon to avoid division by zero
    sharpe_ratio_reward = mean_return / (std_return + 1e-6)

    # Scale down the Sharpe ratio contribution to reduce instability
    sharpe_ratio_reward = 0.01 * sharpe_ratio_reward

    # Final reward with penalty for large deviations and scaling for stability
    return (path[-1] - 0.5 * torch.var(path) + sharpe_ratio_reward) * 10


# Update the train_policy_network function to include SWA and the modified reward function
def train_policy_network_with_swa(data, policy_net, optimizer, num_epochs=100, batch_size=128, epsilon=0.2,
                                  swa_start=75):
    num_batches = len(data) // batch_size
    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(num_batches):
            batch = data[i * batch_size:(i + 1) * batch_size]
            batch_loss = 0
            for path in batch:
                # Prepare input
                path_input = path[:-1]
                adjusted_drift = policy_net(path_input)

                # Add exploration
                if np.random.rand() < epsilon:
                    adjusted_drift += torch.normal(0, 0.1, size=adjusted_drift.size())

                # Calculate Radon-Nikodym derivative for Girsanov's theorem
                new_mu = 0.05 + adjusted_drift
                log_prob = Normal(0.05, 0.2).log_prob(new_mu)

                # Calculate reward using modified reward function
                reward = modified_reward_function(path, path_input + adjusted_drift)

                # Calculate loss and add a scaling factor
                loss = policy_gradient_loss(log_prob, reward)
                grad_penalty = torch.sum(adjusted_drift ** 2)
                loss += 0.00005 * grad_penalty
                batch_loss += loss

            # Backpropagation with increased gradient clipping
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.5)  # Increased clipping value
            optimizer.step()
            total_loss += batch_loss.item() / batch_size

        # Adjust learning rate
        if epoch > swa_start:
            swa_model.update_parameters(policy_net)
            swa_scheduler.step()
        else:
            scheduler.step(total_loss)

        # Print epoch loss
        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss}, Current Learning Rate: {optimizer.param_groups[0]['lr']}")

    # Update BatchNorm statistics for SWA model
    torch.optim.swa_utils.update_bn(data, swa_model)

    return swa_model


# Train the policy network with SWA
swa_policy_net = train_policy_network_with_swa(data, policy_net, optimizer)


# 5. Feynman-Kac Neural Network Solver
class FeynmanKacNN(nn.Module):
    def __init__(self, input_dim):
        super(FeynmanKacNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, input_dim)  # Output is the value function for each time step
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(torch.relu(self.fc2(x)))
        return self.fc3(x)


# Define the Feynman-Kac neural network
feynman_kac_net = FeynmanKacNN(input_dim=num_steps)
fk_optimizer = optim.Adam(feynman_kac_net.parameters(), lr=0.001, weight_decay=1e-4)


# 6. Training the Feynman-Kac Neural Network
def train_feynman_kac_network(data, net, optimizer, num_epochs=100, batch_size=64):  # Increased batch size
    num_batches = len(data) // batch_size
    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(num_batches):
            batch = data[i * batch_size:(i + 1) * batch_size]
            batch_loss = 0
            for path in batch:
                # Prepare input
                path_input = path[:-1]  # Use the path except the final value
                target_value = path[1:]  # Target is the next value in the path

                # Forward pass
                predicted_value = net(path_input)

                # Ensure the predicted_value and target_value are the same length
                predicted_value = predicted_value[:len(target_value)]

                # Loss is the difference between predicted and target values
                loss = nn.MSELoss()(predicted_value, target_value)
                batch_loss += loss

            # Backpropagation with gradient clipping
            optimizer.zero_grad()
            batch_loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += batch_loss.item() / batch_size  # Average over batch size

        # Print epoch loss
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {total_loss}")


# Train the Feynman-Kac network
train_feynman_kac_network(data, feynman_kac_net, fk_optimizer)


# 7. Standard Feynman-Kac Prediction as Benchmark
def feynman_kac_benchmark(num_paths, num_steps, dt, S0, mu, sigma):
    # Initialize benchmark paths
    benchmark_paths = np.zeros((num_paths, num_steps + 1))
    benchmark_paths[:, 0] = S0  # Initial price

    # Loop to generate paths based on the Feynman-Kac formula
    for i in range(num_paths):
        for t in range(1, num_steps + 1):
            dW = np.random.normal(0, np.sqrt(dt))  # Brownian motion increment
            drift = (mu - 0.5 * sigma ** 2) * dt
            diffusion = sigma * dW
            # Update the benchmark path with drift and diffusion terms
            benchmark_paths[i, t] = benchmark_paths[i, t - 1] * np.exp(drift + diffusion)

    return benchmark_paths


# Calculate benchmark predictions
dt = .01
S0 = 100
mu = .05
sigma = .2
benchmark_predictions = feynman_kac_benchmark(num_paths, num_steps, dt, S0, mu, sigma)


# 8. Evaluation and Visualization
def evaluate_model(policy_net, feynman_kac_net, data):
    """
    Evaluate the policy network and Feynman-Kac network on provided data.
    """
    predicted_values = []
    with torch.no_grad():
        if len(data) > 5:
            i = 5
        else:
            i = 1

        for path in data[:i]:  # Evaluate on the first 5 paths for demonstration
            path_input = path[:-1].unsqueeze(0).flatten()  # Flatten and add batch dimension
            adjusted_drift = policy_net(path_input.unsqueeze(0)) * 0.05  # Reduce impact of drift adjustment
            adjusted_path = path_input + adjusted_drift.squeeze(0)  # Apply drift adjustment
            predicted_value = feynman_kac_net(adjusted_path)
            predicted_values.append(predicted_value.numpy())
    return predicted_values


# Evaluate the model
predicted_values = evaluate_model(policy_net, feynman_kac_net, data)


# 9. Ensemble Prediction Function
def ensemble_prediction(policy_net, feynman_kac_net, data, n_ensembles=3):
    """
    Generate ensemble predictions using multiple initializations of policy_net
    to stabilize and improve prediction accuracy.
    """
    ensemble_preds = []
    for _ in range(n_ensembles):
        predictions = []
        with torch.no_grad():
            for path in data[:5]:  # Evaluate on the first 5 paths for demonstration
                path_input = path[:-1]
                adjusted_drift = policy_net(path_input.unsqueeze(0)) * 0.05  # Small impact of drift adjustment
                adjusted_path = path_input + adjusted_drift.squeeze(0)  # Apply drift adjustment
                predicted_value = feynman_kac_net(adjusted_path)
                predictions.append(predicted_value.numpy())
        ensemble_preds.append(predictions)

    # Average predictions across ensembles
    avg_predictions = np.mean(ensemble_preds, axis=0)
    return avg_predictions


# 10. Calculate Ensemble Predictions
predicted_values_ensemble = ensemble_prediction(policy_net, feynman_kac_net, data)


# Function to denormalize data
def denormalize(data, mean, std):
    return data * std + mean


synthetic_paths = []
predicted_paths = []
ensemble_paths = []

for i in range(5):
    synthetic_paths.append(denormalize(data[i], data_mean[i], data_std[i]).numpy())
    predicted_paths.append(denormalize(torch.tensor(predicted_values[i]), data_mean[i], data_std[i]).numpy())
    ensemble_paths.append((denormalize(torch.tensor(predicted_values_ensemble[i]), data_mean[i], data_std[i]).numpy()))

# 11. Plot Synthetic, Benchmark, and Prediction Strategies
for i in range(5):
    plt.figure(figsize=(10, 6))
    plt.plot(synthetic_paths[i], label=f'Synthetic Path {i + 1}')
    plt.plot(range(len(predicted_values[i])), predicted_paths[i], label=f'Feynman-Kac Prediction Path {i + 1}',
             linestyle='--')
    plt.plot(range(len(predicted_values_ensemble[i])), ensemble_paths[i],
             label=f'Ensemble Feynman-Kac Prediction Path {i + 1}', linestyle='-.')  # Adjusted to start from S0 = 100
    plt.plot(range(len(benchmark_predictions[i])), benchmark_predictions[i],
             label=f'Standard Feynman-Kac Benchmark Path {i + 1}', linestyle=':')
    plt.title('Synthetic Asset Price Paths vs Prediction Paths vs Benchmark')
    plt.xlabel('Time Steps')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

#12. Calculate Avg Root Mean Square Error
rmse_network = []
rmse_ensemble = []
rmse_benchmark = []

# Ensure synthetic_paths and predicted_paths have matching lengths for comparison
for i in range(5):
    true_path = synthetic_paths[i][1:]  # Remove the initial value to match the length of predicted paths
    predicted_path_network = predicted_paths[i]
    predicted_path_ensemble = ensemble_paths[i]
    predicted_path_benchmark = benchmark_predictions[i][1:]  # Similarly remove the initial value

    # Calculate RMSE for the neural network prediction and benchmark
    rmse_network.append(root_mean_squared_error(true_path, predicted_path_network))
    rmse_ensemble.append(root_mean_squared_error(true_path, predicted_path_ensemble))
    rmse_benchmark.append(root_mean_squared_error(true_path, predicted_path_benchmark))

avg_rmse_network = np.mean(rmse_network)
avg_rmse_ensemble = np.mean(rmse_ensemble)
avg_rmse_benchmark = np.mean(rmse_benchmark)
print('~~~')
print(f'Average RMSE for Network Predictions: {avg_rmse_network}')
print(f'Average RMSE for Ensemble Predictions: {avg_rmse_ensemble}')
print(f'Average RMSE for Benchmark Predictions: {avg_rmse_benchmark}')


def hedging_with_predictions(path, predicted_path, num_steps, initial_cash=10000, initial_asset=100,
                             transaction_cost=0.01):
    """
    Implement a hedging strategy that uses predictions from Feynman-Kac neural network.
    """
    path_portfolio_value = []
    cash = initial_cash
    asset_holdings = initial_asset

    for t in range(num_steps):
        # Calculate the target position using predicted price
        current_price = path[t + 1] if t + 1 < len(path) else path[-1]
        predicted_price = predicted_path[t]

        threshold = 0.02  # Only trade if predicted and current price difference exceeds 2%

        if abs(predicted_price - current_price) / current_price > threshold:
            if predicted_price > current_price:
                # Buy more if predicted is significantly higher
                amount_to_buy = (predicted_price - current_price) / current_price
                cash -= amount_to_buy * current_price * (1 + transaction_cost)
                asset_holdings += amount_to_buy
            elif predicted_price < current_price:
                # Sell if predicted is significantly lower
                amount_to_sell = (current_price - predicted_price) / current_price
                cash += amount_to_sell * current_price * (1 - transaction_cost)
                asset_holdings -= amount_to_sell

        # Calculate current portfolio value
        portfolio_value = asset_holdings * current_price
        path_portfolio_value.append(portfolio_value)
    return path_portfolio_value


# 13. Use Prediction-Based Hedging Strategy
num_steps = len(predicted_values[0])
hedging_portfolio_values_network = []
hedging_portfolio_values_ensemble = []
hedging_portfolio_values_benchmark = []

for i in range(5):
    hedging_portfolio_values_network.append(hedging_with_predictions(synthetic_paths[i], predicted_paths[i], num_steps))
    hedging_portfolio_values_ensemble.append(hedging_with_predictions(predicted_paths[i], ensemble_paths[i], num_steps))
    hedging_portfolio_values_benchmark.append(
        hedging_with_predictions(synthetic_paths[i], benchmark_predictions[i], num_steps))

# 14. Plot Synthetic, Benchmark, and Hedging Strategies
for i in range(5):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(hedging_portfolio_values_network[i])), hedging_portfolio_values_network[i],
             label=f'Hedging Portfolio Path Neural Network {i + 1}', linestyle='--')
    plt.plot(range(len(hedging_portfolio_values_ensemble[i])), hedging_portfolio_values_ensemble[i],
             label=f'Hedging Portfolio Path Ensemble {i + 1}', linestyle='-.')
    plt.plot(range(len(hedging_portfolio_values_benchmark[i])), hedging_portfolio_values_benchmark[i],
             label=f'Hedging Portfolio Benchmark {i + 1}', linestyle=':')
    plt.title('Synthetic Asset Price Paths vs Feynman Kac Hedging Strategy')
    plt.xlabel('Time Steps')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.show()


#15. Calculate sharpe ratio
def calculate_sharpe_ratio(portfolio_values):
    returns = np.diff(portfolio_values) / portfolio_values[:-1]
    mean_return = np.mean(returns)
    std_return = np.std(returns)
    return mean_return / std_return if std_return != 0 else 0


sharpe_ratios_network = []
sharpe_ratios_ensemble = []
sharpe_ratios_benchmark = []

for i in range(5):
    sharpe_ratios_network.append(calculate_sharpe_ratio(hedging_portfolio_values_network[i]))
    sharpe_ratios_ensemble.append(calculate_sharpe_ratio(hedging_portfolio_values_ensemble[i]))
    sharpe_ratios_benchmark.append(calculate_sharpe_ratio(hedging_portfolio_values_benchmark[i]))

avg_sharpe_network = np.mean(sharpe_ratios_network)
avg_sharpe_ensemble = np.mean(sharpe_ratios_ensemble)
avg_sharpe_benchmark = np.mean(sharpe_ratios_benchmark)

print('~~~')
print(f'Average Sharpe Ratio for Network Hedging: {avg_sharpe_network}')
print(f'Average Sharpe Ratio for Ensemble Hedging: {avg_sharpe_ensemble}')
print(f'Average Sharpe Ratio for Benchmark Hedging: {avg_sharpe_benchmark}')


#16. Calculate final profit/loss
def calculate_final_pnl(portfolio_values):
    return portfolio_values[-1]


final_pnl_network = []
final_pnl_ensemble = []
final_pnl_benchmark = []

for i in range(5):
    final_pnl_network.append(calculate_final_pnl(hedging_portfolio_values_network[i]))
    final_pnl_ensemble.append(calculate_final_pnl(hedging_portfolio_values_ensemble[i]))
    final_pnl_benchmark.append(calculate_final_pnl(hedging_portfolio_values_benchmark[i]))

avg_pnl_network = np.mean(final_pnl_network)
avg_pnl_ensemble = np.mean(final_pnl_ensemble)
avg_pnl_benchmark = np.mean(final_pnl_benchmark)

print('~~~')
print(f'Average Final PnL for Network Hedging: {avg_pnl_network}')
print(f'Average Final PnL for Ensemble: {avg_pnl_ensemble}')
print(f'Average Final PnL for Benchmark Hedging: {avg_pnl_benchmark}')

# ~ Real World Data ~
# Use AAPL as an example to test how real world data will work with the neural network

# 1. Evaluate Neural Network Model
# Normalize Historical data
historical_mean = historical_data.mean(dim=1, keepdim=True)
historical_std = historical_data.std(dim=1, keepdim=True)
historical_data_norm = (historical_data - historical_mean) / historical_std

# Evaluate model using historical data
historical_pred = evaluate_model(policy_net, feynman_kac_net, historical_data_norm)

# 2. Evaluate Benchmark model
# Match number of paths and steps to historical data
num_paths = historical_data.shape[0]
num_steps = historical_data.shape[1] - 1

# Calculate benchmark predictions for Real World data
historical_data_unsq = historical_data  # Unsqueezed data for plotting
historical_data = historical_data.squeeze()  # Removes singleton dimensions

# Calculate variables for benchmark
dt = 1 / 252  # For daily prices
S0 = historical_data[0, 0]
print(f'S0:{S0}')
log_returns = np.log(historical_data[:, 1:] / historical_data[:, :-1])  # Log returns
mu = log_returns.mean() / dt
print(f'mu:{mu}')
sigma = log_returns.std() / np.sqrt(dt)
print(f'sigma:{sigma}')

# Evaluate the benchmark
bench_preds = feynman_kac_benchmark(num_paths, num_steps, dt, S0, mu, sigma)

# 3. Denormalize Data
historical_predicted_path = denormalize(torch.tensor(historical_pred), historical_mean, historical_std).numpy()
historical_predicted_path = historical_predicted_path.flatten().tolist()


# 4. Plot Synthetic, Benchmark, and Prediction Strategies
plt.figure(figsize=(10, 6))
plt.plot(historical_data_unsq[0], label=f'Historical Path')
plt.plot(range(len(historical_pred[0])), historical_predicted_path[:100], label=f'Feynman-Kac Prediction Path',
         linestyle='--')  # Plot first path
plt.plot(range(len(bench_preds[0])), bench_preds[0], label=f'Standard Feynman-Kac Benchmark Path', linestyle=':')
plt.title('Real Asset Price Paths vs Prediction Paths vs Benchmark')
plt.xlabel('Time Steps')
plt.ylabel('Price')
plt.legend()
plt.show()

# 5. Calculate Avg Root Mean Square Error

# Ensure synthetic_paths and predicted_paths have matching lengths for comparison
true_path = historical_data_unsq[0][1:]  # Remove the initial value to match the length of predicted paths
predicted_path_network = historical_predicted_path[:100]
predicted_path_benchmark = bench_preds[0][1:]  # Similarly remove the initial value

# Calculate RMSE for the neural network prediction and benchmark
rmse_network = root_mean_squared_error(true_path, predicted_path_network)
rmse_benchmark = root_mean_squared_error(true_path, predicted_path_benchmark)

avg_rmse_network = np.mean(rmse_network)
avg_rmse_benchmark = np.mean(rmse_benchmark)

print('~~~')
print(f'Average RMSE for Real Network Predictions: {avg_rmse_network}')
print(f'Average RMSE for RwL Benchmark Predictions: {avg_rmse_benchmark}')
