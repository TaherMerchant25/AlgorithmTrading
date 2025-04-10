# Import necessary libraries
import pandas as pd
import numpy as np

# === 1. Heikin-Ashi Candle Calculation ===
def compute_heikin_ashi(df):
    ha_df = df.copy()
    ha_df['HA_Close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_open = [(df['open'].iloc[0] + df['close'].iloc[0]) / 2]
    for i in range(1, len(df)):
        ha_open.append((ha_open[i-1] + ha_df['HA_Close'].iloc[i-1]) / 2)
    ha_df['HA_Open'] = ha_open
    ha_df['HA_High'] = ha_df[['HA_Open', 'HA_Close', 'high']].max(axis=1)
    ha_df['HA_Low'] = ha_df[['HA_Open', 'HA_Close', 'low']].min(axis=1)
    return ha_df

# === 2. ATR and ADX Calculation ===
def add_atr_adx(df):
    df['H-L'] = df['high'] - df['low']
    df['H-PC'] = abs(df['high'] - df['close'].shift(1))
    df['L-PC'] = abs(df['low'] - df['close'].shift(1))
    df['TR'] = df[['H-L', 'H-PC', 'L-PC']].max(axis=1)
    df['ATR'] = df['TR'].rolling(window=14).mean()

    df['+DM'] = np.where((df['high'] - df['high'].shift(1)) > (df['low'].shift(1) - df['low']),
                         np.maximum(df['high'] - df['high'].shift(1), 0), 0)
    df['-DM'] = np.where((df['low'].shift(1) - df['low']) > (df['high'] - df['high'].shift(1)),
                         np.maximum(df['low'].shift(1) - df['low'], 0), 0)

    df['+DI'] = 100 * (df['+DM'].rolling(window=14).mean() / df['ATR'])
    df['-DI'] = 100 * (df['-DM'].rolling(window=14).mean() / df['ATR'])
    df['DX'] = 100 * abs(df['+DI'] - df['-DI']) / (df['+DI'] + df['-DI'])
    df['ADX'] = df['DX'].rolling(window=14).mean()

    df.drop(columns=['H-L', 'H-PC', 'L-PC', 'TR', '+DM', '-DM', '+DI', '-DI', 'DX'], inplace=True)
    return df

# === 3. Final Signal Generation Logic ===
def generate_final_signals(df):
    body_threshold = df['Body'].quantile(0.6)

    conditions = {
        'bullish_ha': df['HA_Close'] > df['HA_Open'],
        'strong_body': df['Body'] > body_threshold,
        'rsi_bullish': df['RSI'] > 0.5,
        'macd_bullish': df['MACD'] > 0.5,
        'adx_trending': df['ADX'] > 20,
        'sentiment_neutral': df['DTW_Sentiment'] == 'Sideways',
    }

    df['Signal'] = 'HOLD'

    # BUY signal
    df.loc[
        conditions['bullish_ha'] &
        conditions['strong_body'] &
        conditions['rsi_bullish'] &
        conditions['macd_bullish'] &
        conditions['adx_trending'] &
        conditions['sentiment_neutral'],
        'Signal'
    ] = 'BUY'

    # SELL signal
    df.loc[
        (~conditions['bullish_ha'] | (df['MACD'] < 0.5)) &
        (df['ADX'] > 20),
        'Signal'
    ] = 'SELL'

    return df

# === 4. Apply Full Pipeline ===
def run_strategy_pipeline(df):
    df_cleaned = df.dropna(subset=['KMeans_Cluster', 'DTW_Cluster', 'KMeans_Sentiment', 'DTW_Sentiment']).copy()
    df_cleaned = add_atr_adx(df_cleaned)
    df_cleaned_reset = df_cleaned.reset_index(drop=True)
    df_ha = compute_heikin_ashi(df_cleaned_reset)
    df_signaled = generate_final_signals(df_ha)
    return df_signaled

# Example usage:
# df = pd.read_csv("btc_hourly_clustered_extended.csv")
# result_df = run_strategy_pipeline(df)
# print(result_df['Signal'].value_counts())