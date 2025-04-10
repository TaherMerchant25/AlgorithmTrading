
from binance.client import Client
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np
import ta

# 1. Fetch 1-hour BTC/USDT data
client = Client()
klines = client.get_historical_klines(
    "BTCUSDT",
    Client.KLINE_INTERVAL_1HOUR,
    "1 Jan, 2020",
    "31 Dec, 2024"
)

# 2. Create DataFrame
columns = [
    'Open Time', 'Open', 'High', 'Low', 'Close', 'Volume',
    'Close Time', 'Quote Asset Volume', 'Number of Trades',
    'Taker Buy Base Asset Volume', 'Taker Buy Quote Asset Volume', 'Ignore'
]
data = pd.DataFrame(klines, columns=columns)

# 3. Convert timestamps
data['Open Time'] = pd.to_datetime(data['Open Time'], unit='ms')
data['Close Time'] = pd.to_datetime(data['Close Time'], unit='ms')

# 4. Convert to numeric
for col in ['Open', 'High', 'Low', 'Close', 'Volume',
            'Quote Asset Volume', 'Taker Buy Base Asset Volume', 
            'Taker Buy Quote Asset Volume']:
    data[col] = data[col].astype(float)

# Technical Indicators
data['RSI'] = ta.momentum.RSIIndicator(close=data['Close'], window=14).rsi()

macd = ta.trend.MACD(close=data['Close'])
data['MACD'] = macd.macd_diff()

data['EMA_12'] = ta.trend.EMAIndicator(close=data['Close'], window=12).ema_indicator()
data['EMA_26'] = ta.trend.EMAIndicator(close=data['Close'], window=26).ema_indicator()
data['EMA_diff'] = data['EMA_12'] - data['EMA_26']

# Feature Engineering for 1h
data['Return'] = (data['Close'] - data['Open']) / data['Open']
data['Volatility'] = (data['High'] - data['Low']) / data['Open']

data['Momentum_3h'] = data['Close'].pct_change(periods=3)                            # 3 hours
data['Rolling_Return_Mean'] = data['Return'].rolling(window=6).mean()               # 6 hours
data['Rolling_Return_Std'] = data['Return'].rolling(window=6).std()                 # 6 hours
data['Volume_ZScore'] = (data['Volume'] - data['Volume'].rolling(12).mean()) / data['Volume'].rolling(12).std()  # 12 hours
data['Price_Range_Ratio'] = (data['High'] - data['Low']) / data['Close']
data['Candle_Body_Ratio'] = (data['Close'] - data['Open']) / (data['High'] - data['Low'] + 1e-6)
data['Prev_Close_Change'] = data['Close'].pct_change(periods=1)

# Drop NaNs
data = data.dropna().reset_index(drop=True)

# Feature list
feature_cols = [
    'Return', 'Volatility', 'Momentum_3h', 'Rolling_Return_Mean',
    'Rolling_Return_Std', 'Volume_ZScore', 'Price_Range_Ratio',
    'Candle_Body_Ratio', 'Prev_Close_Change',
    'RSI', 'MACD', 'EMA_diff'
]

# Clustering
features = data[feature_cols]
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)

kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(features_scaled)

# Label clusters by avg return
centroids_scaled = kmeans.cluster_centers_
centroids = scaler.inverse_transform(centroids_scaled)

cluster_labels = {}
for i, centroid in enumerate(centroids):
    avg_return = centroid[0]
    if avg_return > 0.005:
        cluster_labels[i] = "Bullish"
    elif avg_return < -0.005:
        cluster_labels[i] = "Bearish"
    else:
        cluster_labels[i] = "Sideways"
    print(f"Cluster {i}: Avg Return = {avg_return:.4f} → {cluster_labels[i]}")

data['Market_Type'] = data['Cluster'].map(cluster_labels)

# Preview
print(data[['Open Time', 'Return', 'Volatility', 'Market_Type']].head(10))

data.to_csv('1hr.csv', index=False)
