import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from binance.client import Client
from ta.momentum import RSIIndicator
from ta.trend import MACD
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax

# Binance API client (no keys needed for public market data)
client = Client()

# Parameters
symbol = "BTCUSDT"
interval = Client.KLINE_INTERVAL_1DAY
limit = 300  # up to 1000 days of data

# Fetch daily candlestick data
klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)

# Convert to DataFrame
df = pd.DataFrame(klines, columns=[
    'timestamp', 'open', 'high', 'low', 'close', 'volume',
    'close_time', 'quote_asset_volume', 'number_of_trades',
    'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
])

# Preprocess columns
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
df.set_index('timestamp', inplace=True)
df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)

# Calculate RSI and MACD
df['RSI'] = RSIIndicator(close=df['close'], window=14).rsi()
macd = MACD(close=df['close'])
df['MACD'] = macd.macd()

# Remove NaN values
df.dropna(inplace=True)

# Build sequences using a sliding window
window_size = 14  # 14-day time windows
sequences = []
timestamps = []

for i in range(len(df) - window_size + 1):
    rsi_seq = df['RSI'].values[i:i+window_size]
    macd_seq = df['MACD'].values[i:i+window_size]
    combined_seq = np.vstack((rsi_seq, macd_seq)).T  # shape: (window_size, 2)
    sequences.append(combined_seq)
    timestamps.append(df.index[i + window_size - 1])

sequences = np.array(sequences)

# Normalize the sequences
scaler = TimeSeriesScalerMinMax()
sequences_scaled = scaler.fit_transform(sequences)

# Apply TimeSeriesKMeans with DTW
n_clusters = 3
model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=42)
labels = model.fit_predict(sequences_scaled)

# Create clustered DataFrame
cluster_df = pd.DataFrame({
    'timestamp': timestamps,
    'Cluster': labels
})
cluster_df.set_index('timestamp', inplace=True)

# Merge cluster info into main DataFrame
df_clustered = df.copy()
df_clustered = df_clustered.merge(cluster_df, how='left', left_index=True, right_index=True)

# Plot clusters
plt.figure(figsize=(12, 6))
for cluster in range(n_clusters):
    points = df_clustered[df_clustered['Cluster'] == cluster]
    plt.scatter(points['RSI'], points['MACD'], label=f'Cluster {cluster}', alpha=0.6)

plt.xlabel("RSI")
plt.ylabel("MACD")
plt.title("DTW + KMeans Clustering on BTCUSDT (1D Interval)")
plt.legend()
plt.grid(True)
plt.show()

# Export to CSV
df_clustered.to_csv("btc_daily_clustered_data.csv")
print("✅ Clustered data exported to 'btc_daily_clustered_data.csv'")
