# **Stochastic Control with Machine Learning for Optimal Measure Change**  

This project applies **reinforcement learning** and **stochastic control** to optimize **measure change** via **Girsanov’s theorem**. It incorporates:  
- A **policy network** for drift adjustments  
- A **Feynman-Kac solver** for path predictions  
- **Stochastic Weight Averaging (SWA)** for stability  
- **Real-world data benchmarking** against synthetic models  
- **Ensemble predictions** for improved accuracy  
- **Hedging strategies** based on predicted paths  

This framework is useful for **quantitative finance, risk management, and trading strategies**.  

---

## **Installation**  

### **1. Clone the Repository**  
```bash
git clone https://github.com/yourusername/stochastic-control-ml.git
cd stochastic-control-ml
```

### **2. Create and Activate Virtual Environment (Recommended)**  
```bash
python -m venv venv
source venv/bin/activate  # For macOS/Linux
venv\Scripts\activate     # For Windows
```

### **3. Install Dependencies**  
```bash
pip install -r requirements.txt
```

### **4. Dependencies**  
The following libraries are required:  
- `torch` (PyTorch) – Neural networks and reinforcement learning  
- `numpy` – Data handling  
- `matplotlib` – Visualization  
- `yfinance` – Fetching real-world financial data  
- `sklearn` – Evaluation metrics  

To install them manually:  
```bash
pip install torch numpy matplotlib yfinance scikit-learn
```

---

## **How It Works**  

### **1. Data Generation**  
- **Synthetic Data**: Simulated Monte Carlo paths using a **geometric Brownian motion (GBM)** model.  
- **Real-World Data**: Historical stock prices fetched via **Yahoo Finance API**.  

### **2. Policy Network (Reinforcement Learning Agent)**  
- Predicts drift adjustments based on past price movements.  
- Uses an **exploration-exploitation** strategy.  
- Optimized via **policy gradient loss and Radon-Nikodym derivatives** for Girsanov’s theorem.  

### **3. Feynman-Kac Neural Network**  
- Estimates the expected future value of paths.  
- Provides a **benchmark prediction** based on stochastic control.  

### **4. Training Loop (Policy Gradient Optimization)**  
- **Rolling Window Training**: Learns on past data but predicts future prices.  
- **Exploration Mechanism**: Introduces controlled noise to encourage learning.  
- **Gradient Penalty**: Prevents overfitting by regularizing drift adjustments.  

### **5. Stochastic Weight Averaging (SWA)**  
- Stabilizes training by averaging model parameters.  
- Activated after a set number of epochs to smooth weight updates.  

### **6. Evaluation & Benchmarking**  
- **Predicts future asset prices** using the trained policy network.  
- **Compares predictions** against standard Feynman-Kac solutions and real-world data.  
- **Measures performance** using RMSE (Root Mean Squared Error).  

### **7. Hedging Strategy Simulation**  
- Simulates a **trading strategy** based on predicted price movements.  
- **Adapts portfolio allocation** dynamically based on forecasted trends.  
- Evaluates **Sharpe ratio** and **PnL (Profit & Loss)** to assess performance.  

---

## **Usage**  

### **Run the Training Script**  
```bash
python train.py
```
This will:  
✔ Generate synthetic data  
✔ Train the **policy network** and **Feynman-Kac model**  
✔ Apply **stochastic weight averaging (SWA)**  
✔ Benchmark performance  

### **Run Real-World Data Evaluation**  
```bash
python test.py
```
This will:  
✔ Load **real-world stock price data**  
✔ Predict future movements using trained networks  
✔ Compare results with standard **Feynman-Kac solutions**  

### **Run Hedging Simulations**  
```bash
python hedging.py
```
This will:  
✔ Simulate trading using **forecasted prices**  
✔ Compute **portfolio performance**  
✔ Calculate **Sharpe ratio and final PnL**  

---

## **Results & Visualization**  

1. **Synthetic vs. Predicted vs. Benchmark Paths**  
<img width="1269" alt="synthetic_vs_benchmark" src="https://github.com/user-attachments/assets/66039b19-fa64-4433-8df5-2948470bf8f8" />

2. **Real-World Predictions vs. Benchmark**  
<img width="1161" alt="real_data_predictions png" src="https://github.com/user-attachments/assets/2329e863-4834-4c1e-9ed9-86be4a195c65" />

3. **Hedging Portfolio Performance**  
<img width="1228" alt="hedging_portfolio" src="https://github.com/user-attachments/assets/92aaf0f4-78f8-4c08-b1fc-2f84e7024adc" />

---

## **Future Improvements**  
🔹 Extend the **policy network** for multi-asset portfolios.  
🔹 Improve exploration using **adaptive ε-greedy methods**.  
🔹 Integrate **LSTMs or transformers** for better sequential learning.  
🔹 Implement **Bayesian optimization** for hyperparameter tuning.  

---

## **Contributors**  
👤 **David Abramson** - [GitHub](https://github.com/davidonthethird)  

---

## **License**  
This project is licensed under the **MIT License**.  
