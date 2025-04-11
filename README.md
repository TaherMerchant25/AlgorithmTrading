# 📈 Zelta Labs Algo Trading Hackathon – Multi-Strategy Trading System

Welcome to our submission for the **Zelta Labs Algo Trading Hackathon**. This project showcases a **multi-strategy framework** built using Python, designed for backtesting and live deployment on platforms like Binance, Kite, or Fyers.

## 🚀 Strategies Included

| Strategy Name | Description |
|---------------|-------------|
| 📉 **Heikin-Ashi Trend Catcher** | Uses Heikin-Ashi candlesticks to identify smoothed trends with minimal noise |
| ⚡ **MACD + RSI Momentum Burst** | Combines momentum oscillators to catch strong entries with confirmation |
| 📈 **EMA Crossover System** | Classic fast/slow EMA crossover with volatility filtering |
| 🌀 **Ichimoku Cloud Bias** | Uses cloud direction and crossovers to detect long-term biases |
| 🔍 **ADX Strength Filter** | Confirms trend quality before entering trades |
| 🎯 **Volume Spike Detection** | Looks for unusual volume behavior with breakout confirmation |
| 📊 **Signal Clustering** | Aggregates technical indicators for probabilistic scoring |

---

## 📁 Project Structure
```
├── data/ # Historical and live market data ├── strategies/ │ ├── heikin_ashi_strategy.py # Heikin-Ashi trend logic │ ├── momentum_strategy.py # MACD + RSI based logic │ ├── ema_crossover.py # EMA crossover + volatility filter │ ├── ichimoku_strategy.py # Ichimoku Cloud strategy │ └── adx_filter.py # Trend quality filter ├── utils/ │ ├── backtest.py # Backtesting engine │ ├── signal_cluster.py # Cluster signals from multiple strategies │ └── risk_manager.py # Dynamic SL/TP, position sizing ├── main.py # Strategy selection and execution ├── requirements.txt # Python dependencies └── README.md # You’re reading it!
```

---

## 🧠 How It Works

Each strategy generates buy/sell signals on a per-candle basis. Signals are then:
- Combined using a **voting or clustering system**
- Filtered by **volatility and ADX**
- Passed through a **risk manager** that adjusts capital allocation

---

## 🔧 Setup Instructions

### 1. Clone the Repo

```bash
git clone https://github.com/your-username/zelta-hackathon-strategies.git
cd zelta-hackathon-strategies
```


