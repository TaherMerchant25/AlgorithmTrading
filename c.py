import pandas as pd
import numpy as np
from binance.client import Client
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import AverageTrueRange
import matplotlib.pyplot as plt

def calculate_heikin_ashi(df):
    ha_df = df.copy()
    ha_df['HA_Close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_df['HA_Open'] = df['open'].copy()
    for i in range(1, len(df)):
        ha_df.iloc[i, ha_df.columns.get_loc('HA_Open')] = (ha_df['HA_Open'].iloc[i-1] + ha_df['HA_Close'].iloc[i-1]) / 2
    ha_df['HA_High'] = ha_df[['high', 'HA_Open', 'HA_Close']].max(axis=1)
    ha_df['HA_Low'] = ha_df[['low', 'HA_Open', 'HA_Close']].min(axis=1)
    return ha_df

def prepare_data(symbol="BTCUSDT", interval="1h", limit=1000, use_clustering=True):
    try:
        df = pd.read_csv('btc_hourly_clustered_extended.csv')
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df = df.astype({col: float for col in ['open', 'high', 'low', 'close', 'volume']})
        print("Successfully loaded data from CSV")
    except Exception as e:
        print(f"Couldn't load data from CSV: {e}")
        print("Fetching data from Binance API")
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
    
    df['EMA5'] = df['close'].ewm(span=5, adjust=False).mean()
    df['EMAx'] = df['close'].ewm(span=14, adjust=False).mean()
    atr_indicator = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['ATR'] = atr_indicator.average_true_range()
    adx_indicator = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['adx'] = adx_indicator.adx()
    df['WeeklyReturns'] = df['close'].pct_change(5).fillna(0)
    df = calculate_heikin_ashi(df)
    df['Position'] = 0
    df['Signal'] = 0

    if use_clustering and 'KMeans_Cluster' in df.columns:
        required_cols = ['KMeans_Cluster', 'DTW_Cluster', 'KMeans_Sentiment', 'DTW_Sentiment']
        for col in required_cols:
            if col not in df.columns:
                print(f"Warning: Expected column {col} not found. Adding with default values.")
                df[col] = None
        df[required_cols] = df[required_cols].ffill()
        print("Successfully processed clustering data")
    elif use_clustering:
        print("Clustering columns not found in data")
    df = df.reset_index()
    return df

def HA_and_Shortsell_with_Clusters(df, atr_multiplier=4, stoploss=0):
    if 'Position' not in df.columns:
        df['Position'] = 0
    if 'Signal' not in df.columns:
        df['Signal'] = 0

    stop_loss = 0
    for i in range(200, len(df)-1):
        prev_ha_close = df['HA_Close'].iloc[i-1]
        curr_ha_close = df['HA_Close'].iloc[i]
        prev_ha_open = df['HA_Open'].iloc[i-1]
        curr_ha_open = df['HA_Open'].iloc[i]
        curr_atr = df['ATR'].iloc[i]
        kmeans_sentiment = df['KMeans_Sentiment'].iloc[i] if 'KMeans_Sentiment' in df.columns else None
        dtw_sentiment = df['DTW_Sentiment'].iloc[i] if 'DTW_Sentiment' in df.columns else None
        sentiment_score = 0
        if kmeans_sentiment == 'Bullish':
            sentiment_score += 1
        elif kmeans_sentiment == 'Bearish':
            sentiment_score -= 1
        if dtw_sentiment == 'Bullish':
            sentiment_score += 1
        elif dtw_sentiment == 'Bearish':
            sentiment_score -= 1

        if (df['close'].iloc[i] < df['EMAx'].iloc[i-1] and 
            df['low'].iloc[i-1] > df['EMA5'].iloc[i-1] and 
            df['close'].iloc[i] < df['low'].iloc[i-1] and 
            df['Position'].iloc[i] == 0 and
            (sentiment_score <= 0 or 'KMeans_Sentiment' not in df.columns)):
            df.loc[i+1, 'Position'] = -1
            df.loc[i, 'Signal'] = -df.loc[i, 'Position']
            stop_loss = df['high'].iloc[i-1] + 2*df['ATR'].iloc[i-1]

        elif (df['close'].iloc[i] > df['EMAx'].iloc[i-1] and 
              prev_ha_close > prev_ha_open and 
              curr_ha_close > curr_ha_open and 
              df['adx'].iloc[i] < 25 and 
              df['Position'].iloc[i] != 1 and
              (sentiment_score >= 0 or 'KMeans_Sentiment' not in df.columns)):
            df.loc[i+1, 'Position'] = 1
            df.loc[i, 'Signal'] = 1 - df.loc[i, 'Position']
            stop_loss = df['high'].iloc[i] - curr_atr * (atr_multiplier+1)

        elif (df['close'].iloc[i] > df['EMAx'].iloc[i-1] and 
              prev_ha_close > prev_ha_open and 
              curr_ha_close < curr_ha_open and 
              df['adx'].iloc[i] < 25 and 
              df['Position'].iloc[i] != -1 and
              (sentiment_score <= 0 or 'KMeans_Sentiment' not in df.columns)):
            df.loc[i+1, 'Position'] = -1
            df.loc[i, 'Signal'] = -1 - df.loc[i, 'Position']
            stop_loss = df['low'].iloc[i] + curr_atr * (atr_multiplier-1)

        elif (df['adx'].iloc[i] > 60 and 
              df['Position'].iloc[i] != 1 and 
              df['WeeklyReturns'].iloc[i] < 0 and
              (sentiment_score > 0 or 'KMeans_Sentiment' not in df.columns)):
            df.loc[i+1, 'Position'] = 1
            df.loc[i, 'Signal'] = 1 - df.loc[i, 'Position']
            stop_loss = df['high'].iloc[i] - curr_atr * (atr_multiplier+1)

        elif (df['adx'].iloc[i] > 60 and 
              df['Position'].iloc[i] != -1 and 
              df['WeeklyReturns'].iloc[i] > 0 and
              (sentiment_score < 0 or 'KMeans_Sentiment' not in df.columns)):
            df.loc[i+1, 'Position'] = -1
            df.loc[i, 'Signal'] = -1 - df.loc[i, 'Position']
            stop_loss = df['low'].iloc[i] + curr_atr * (atr_multiplier-1)

        elif ('KMeans_Sentiment' in df.columns and
              ((df['Position'].iloc[i] == 1 and sentiment_score < -1) or 
               (df['Position'].iloc[i] == -1 and sentiment_score > 1))):
            df.loc[i+1, 'Position'] = 0
            df.loc[i, 'Signal'] = -df.loc[i, 'Position']

        else:
            if df['Position'].iloc[i] == 1:
                df.loc[i+1, 'Position'] = 1
                if df['low'].iloc[i] < stop_loss:
                    df.loc[i+1, 'Position'] = 0
                    df.loc[i, 'Signal'] = -1
                else:
                    stop_loss = max(stop_loss, df['high'].iloc[i] - curr_atr * (atr_multiplier+1))
            elif df['Position'].iloc[i] == -1:
                df.loc[i+1, 'Position'] = -1
                if df['high'].iloc[i] > stop_loss:
                    df.loc[i+1, 'Position'] = 0
                    df.loc[i, 'Signal'] = 1
                else:
                    stop_loss = min(stop_loss, df['low'].iloc[i] + curr_atr * (atr_multiplier-1))
            else:
                df.loc[i+1, 'Position'] = 0

    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    return df

def analyze_strategy_performance(df):
    df['Strategy_Returns'] = df['Position'].shift(1) * df['close'].pct_change()
    df['Cumulative_Returns'] = (1 + df['Strategy_Returns'].fillna(0)).cumprod() - 1
    total_return = df['Cumulative_Returns'].iloc[-1]
    sharpe_ratio = df['Strategy_Returns'].mean() / max(df['Strategy_Returns'].std(), 1e-8) * np.sqrt(252)
    max_drawdown = (df['Cumulative_Returns'] - df['Cumulative_Returns'].cummax()).min()
    win_count = len(df[df['Strategy_Returns'] > 0])
    total_trades = len(df[df['Strategy_Returns'] != 0])
    win_rate = win_count / max(total_trades, 1)

    # Time to Recovery (double digits)
    drawdown_series = df['Cumulative_Returns'] - df['Cumulative_Returns'].cummax()
    recovery_times = []
    peak = df['Cumulative_Returns'].cummax()
    for i in range(len(df)):
        if drawdown_series.iloc[i] < -0.10:
            for j in range(i+1, len(df)):
                if df['Cumulative_Returns'].iloc[j] >= peak.iloc[i]:
                    recovery_times.append((df.index[j] - df.index[i]).total_seconds() / 3600)
                    break

    avg_time_to_recovery = np.mean(recovery_times) if recovery_times else 0

    # Quarterly/Yearly Profitability
    df['Quarter'] = df.index.to_series().dt.to_period("Q")
    df['Year'] = df.index.to_series().dt.year
    quarterly_returns = df.groupby('Quarter')['Strategy_Returns'].sum()
    yearly_returns = df.groupby('Year')['Strategy_Returns'].sum()

    performance = {
        'Total Return': f"{total_return:.2%}",
        'Sharpe Ratio': f"{sharpe_ratio:.2f}",
        'Max Drawdown': f"{max_drawdown:.2%}",
        'Win Rate': f"{win_rate:.2%}",
        'Total Trades': total_trades,
        'Avg Time to Recovery (hrs)': f"{avg_time_to_recovery:.2f}",
        'Quarterly Profitability': quarterly_returns.round(4).to_dict(),
        'Yearly Profitability': yearly_returns.round(4).to_dict()
    }
    return performance

def plot_strategy_results(df, title="Strategy Performance"):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), gridspec_kw={'height_ratios': [2, 1]})
    ax1.plot(df.index, df['close'], label='Price', color='black', alpha=0.5)
    buy_signals = df[df['Signal'] == 1]
    if not buy_signals.empty:
        ax1.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Buy')
    sell_signals = df[df['Signal'] == -1]
    if not sell_signals.empty:
        ax1.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='Sell')
    ax1.set_title(f'{title} - Price and Signals')
    ax1.set_ylabel('Price')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(df.index, df['Cumulative_Returns'], label='Strategy Returns', color='blue')
    ax2.plot(df.index, df['close'].pct_change().fillna(0).cumsum(), label='Buy & Hold', color='grey', alpha=0.5)
    ax2.set_title('Equity Curve')
    ax2.set_ylabel('Returns')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('strategy_results.png')
    plt.show()

def run_strategy_pipeline():
    df = prepare_data(symbol="BTCUSDT", interval="1h", limit=1000, use_clustering=True)
    df = HA_and_Shortsell_with_Clusters(df, atr_multiplier=4)
    performance = analyze_strategy_performance(df)

    print("\n===== STRATEGY PERFORMANCE =====")
    for metric, value in performance.items():
        print(f"{metric}: {value}")
    
    plot_strategy_results(df, title="Enhanced HA Strategy with Clustering")
    df.to_csv("strategy_results_with_clusters.csv")
    return df, performance

if __name__ == "__main__":
    df, performance = run_strategy_pipeline()
