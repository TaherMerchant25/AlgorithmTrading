import pandas as pd
import numpy as np
from binance.client import Client
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.preprocessing import MinMaxScaler
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax

# ========== FETCH BINANCE DATA ==========
def fetch_binance_data(symbol="BTCUSDT", interval="1d", limit=300):
    client = Client()
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    return df

# ========== FEATURE ENGINEERING ==========
def add_features(df):
    df['RSI'] = RSIIndicator(close=df['close'], window=14).rsi()
    macd = MACD(close=df['close'])
    df['MACD'] = macd.macd()

    df['HL_Volatility'] = df['high'] - df['low']
    df['Body'] = df['close'] - df['open']
    df['Volume_Change'] = df['volume'].pct_change().fillna(0)
    return df.dropna()

# ========== SCALING ==========
def scale_features(df, cols_to_scale):
    scaler = MinMaxScaler()
    df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
    return df

# ========== SEQUENCE CREATION ==========
def create_sequences(df, window=14, features=None):
    sequences = []
    timestamps = []
    for i in range(len(df) - window + 1):
        window_df = df[features].iloc[i:i+window].values
        sequences.append(window_df)
        timestamps.append(df.index[i + window - 1])
    return np.array(sequences), timestamps

# ========== CLUSTERING ==========
def cluster_sequences(sequences, n_clusters=3):
    scaler = TimeSeriesScalerMinMax()
    sequences_scaled = scaler.fit_transform(sequences)
    model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=42)
    labels = model.fit_predict(sequences_scaled)
    return labels

# ========== SHARPE METRIC ==========
def compute_sharpe_ratio(df, risk_free_rate=0.0):
    daily_returns = df['close'].pct_change().dropna()
    avg_return = daily_returns.mean()
    std_dev = daily_returns.std()
    sharpe = (avg_return - risk_free_rate) / std_dev if std_dev != 0 else 0
    return sharpe, avg_return, std_dev

# ========== MAIN PIPELINE ==========
def main():
    df = fetch_binance_data(interval=Client.KLINE_INTERVAL_1DAY)
    df = add_features(df)

    # Features to use
    feature_cols = ['RSI', 'MACD', 'HL_Volatility', 'Body', 'Volume_Change']
    df = scale_features(df, feature_cols)

    sequences, timestamps = create_sequences(df, window=14, features=feature_cols)
    labels = cluster_sequences(sequences, n_clusters=3)

    # Merge cluster labels
    cluster_df = pd.DataFrame({'timestamp': timestamps, 'Cluster': labels})
    cluster_df.set_index('timestamp', inplace=True)
    df_clustered = df.copy()
    df_clustered = df_clustered.merge(cluster_df, how='left', left_index=True, right_index=True)

    # Assign sentiment labels based on avg returns
    cluster_labels = {}
    print("\n📊 Cluster Performance:")
    for cluster in sorted(df_clustered['Cluster'].dropna().unique()):
        cluster_data = df_clustered[df_clustered['Cluster'] == cluster]
        sharpe, avg_return, volatility = compute_sharpe_ratio(cluster_data)

        # Sentiment labeling
        if avg_return > 0.002:
            label = "Bullish"
        elif avg_return < -0.002:
            label = "Bearish"
        else:
            label = "Sideways"

        cluster_labels[cluster] = label
        print(f"Cluster {int(cluster)} | Sharpe: {sharpe:.2f} | Return: {avg_return:.4f} | Volatility: {volatility:.4f} | Label: {label}")

    df_clustered['Sentiment'] = df_clustered['Cluster'].map(cluster_labels)

    # Export
    df_clustered.to_csv("btc_clustered_labeled.csv")
    print("\n✅ Exported to btc_clustered_labeled.csv")

if __name__ == "__main__":
    main()
