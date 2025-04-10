import pandas as pd
import numpy as np
from binance.client import Client
from ta.momentum import RSIIndicator
from ta.trend import MACD
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMinMax

# ========== FETCH BINANCE DATA ==========
def fetch_binance_data(symbol="BTCUSDT", interval="1h", limit=500):
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

# ========== EXTENDED FEATURE ENGINEERING ==========
def add_features(df):
    # RSI & MACD
    df['RSI'] = RSIIndicator(close=df['close'], window=14).rsi()
    macd = MACD(close=df['close'])
    df['MACD'] = macd.macd()

    # Price-based features
    df['HL_Volatility'] = df['high'] - df['low']
    df['Body'] = df['close'] - df['open']
    df['Volume_Change'] = df['volume'].pct_change().fillna(0)
    df['Price_Change'] = df['close'].diff().fillna(0)
    df['Pct_Change'] = df['close'].pct_change().fillna(0)
    df['Upper_Shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['Lower_Shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    
    # Trend indicators
    df['SMA_14'] = df['close'].rolling(window=14).mean()
    df['EMA_14'] = df['close'].ewm(span=14, adjust=False).mean()
    df['Momentum'] = df['close'] - df['close'].shift(14)
    df['Rolling_Volatility_14'] = df['close'].pct_change().rolling(14).std()

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

# ========== CLUSTERING METHODS ==========
def cluster_kmeans_flat(sequences, n_clusters=3):
    flat_sequences = sequences.reshape(sequences.shape[0], -1)
    model = KMeans(n_clusters=n_clusters, random_state=42)
    labels = model.fit_predict(flat_sequences)
    return labels

def cluster_dtw(sequences, n_clusters=3):
    scaler = TimeSeriesScalerMinMax()
    sequences_scaled = scaler.fit_transform(sequences)
    model = TimeSeriesKMeans(n_clusters=n_clusters, metric="dtw", random_state=42)
    labels = model.fit_predict(sequences_scaled)
    return labels

# ========== SHARPE METRIC ==========
def compute_sharpe_ratio(df, risk_free_rate=0.0):
    returns = df['close'].pct_change().dropna()
    avg_return = returns.mean()
    std_dev = returns.std()
    sharpe = (avg_return - risk_free_rate) / std_dev if std_dev != 0 else 0
    return sharpe, avg_return, std_dev

# ========== CLUSTER SENTIMENT LABELING ==========
def assign_sentiment_labels(df, cluster_column):
    cluster_labels = {}
    print(f"\n📊 Cluster Performance ({cluster_column}):")
    for cluster in sorted(df[cluster_column].dropna().unique()):
        cluster_data = df[df[cluster_column] == cluster]
        sharpe, avg_return, volatility = compute_sharpe_ratio(cluster_data)

        if avg_return > 0.001:
            label = "Bullish"
        elif avg_return < -0.001:
            label = "Bearish"
        else:
            label = "Sideways"

        cluster_labels[cluster] = label
        print(f"Cluster {int(cluster)} | Sharpe: {sharpe:.2f} | Return: {avg_return:.5f} | Volatility: {volatility:.5f} | Label: {label}")
    return df[cluster_column].map(cluster_labels)

# ========== MAIN PIPELINE ==========
def main():
    df = fetch_binance_data(interval=Client.KLINE_INTERVAL_1HOUR)
    df = add_features(df)

    # Updated feature list
    feature_cols = [
        'RSI', 'MACD', 'HL_Volatility', 'Body', 'Volume_Change',
        'Price_Change', 'Pct_Change', 'Upper_Shadow', 'Lower_Shadow',
        'SMA_14', 'EMA_14', 'Momentum', 'Rolling_Volatility_14'
    ]

    df = scale_features(df, feature_cols)

    sequences, timestamps = create_sequences(df, window=14, features=feature_cols)

    # KMeans clustering
    kmeans_labels = cluster_kmeans_flat(sequences, n_clusters=3)
    kmeans_df = pd.DataFrame({'timestamp': timestamps, 'KMeans_Cluster': kmeans_labels}).set_index('timestamp')

    # DTW clustering
    dtw_labels = cluster_dtw(sequences, n_clusters=3)
    dtw_df = pd.DataFrame({'timestamp': timestamps, 'DTW_Cluster': dtw_labels}).set_index('timestamp')

    # Merge
    df_clustered = df.copy()
    df_clustered = df_clustered.merge(kmeans_df, how='left', left_index=True, right_index=True)
    df_clustered = df_clustered.merge(dtw_df, how='left', left_index=True, right_index=True)

    # Sentiment labels
    df_clustered['KMeans_Sentiment'] = assign_sentiment_labels(df_clustered, 'KMeans_Cluster')
    df_clustered['DTW_Sentiment'] = assign_sentiment_labels(df_clustered, 'DTW_Cluster')

    # Export
    df_clustered.to_csv("btc_hourly_clustered_extended1.csv")
    print("\n Exported to btc_hourly_clustered_extended1.csv")

if __name__ == "__main__":
    main()
