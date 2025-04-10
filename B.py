import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# import yfinance as yf
from datetime import datetime

df = pd.read_csv(r'btc_hourly_clustered_extended.csv') 

def calculate_atr(df, period):
    """Calculate the Average True Range indicator"""
    df_copy = df.copy()
    df_copy['High-Low'] = abs(df_copy['high'] - df_copy['low'])
    df_copy['High-PreviousClose'] = abs(df_copy['high'] - df_copy['close'].shift(1))
    df_copy['Low-PreviousClose'] = abs(df_copy['low'] - df_copy['close'].shift(1))
    df_copy['TrueRange'] = df_copy[['High-Low', 'High-PreviousClose', 'Low-PreviousClose']].max(axis=1, skipna=False)
    df_copy['ATR'] = df_copy['TrueRange'].ewm(com=period, min_periods=period).mean()
    return df_copy['ATR']

def calculate_adx(df, lookback):
    """Calculate Average Directional Index (ADX)"""
    high = df['high']
    low = df['low']
    close = df['close']
    
    plus_dm = high.diff()
    minus_dm = low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0

    tr1 = pd.DataFrame(high - low)
    tr2 = pd.DataFrame(abs(high - close.shift(1)))
    tr3 = pd.DataFrame(abs(low - close.shift(1)))
    frames = [tr1, tr2, tr3]
    tr = pd.concat(frames, axis=1, join='inner').max(axis=1)
    atr = tr.rolling(lookback).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1/lookback).mean() / atr)
    minus_di = abs(100 * (minus_dm.ewm(alpha=1/lookback).mean() / atr))
    dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
    adx = ((dx.shift(1) * (lookback - 1)) + dx) / lookback
    adx_smooth = adx.ewm(alpha=1/lookback).mean()
    return plus_di, minus_di, adx_smooth

def calculate_heikin_ashi(df):
    """Calculate Heikin-Ashi candles from regular OHLC data"""
    ha_df = df.copy()
    
    # Calculate Heikin-Ashi candles
    ha_df['HA_Close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_df['HA_Open'] = (df['open'].shift(1) + df['close'].shift(1)) / 2
    ha_df['HA_High'] = df[['high', 'low', 'HA_Open', 'HA_Close']].max(axis=1)
    ha_df['HA_Low'] = df[['high', 'low', 'HA_Open', 'HA_Close']].min(axis=1)
    
    return ha_df

def fetch_data(symbol, start_date, end_date, interval="1d"):
    """Fetch historical price data from Yahoo Finance"""
    try:
        data = df
        df = data.reset_index()
        # Convert column names to lowercase for consistency
        df.columns = [col.lower() for col in df.columns]
        if 'date' in df.columns:
            df = df.rename(columns={'date': 'timestamp'})
        return df
    except Exception as e:
        print(f"Error fetching data: {e}")
        return None

def prepare_data(df, atr_period=14, adx_short_period=15, adx_long_period=250):
    """
    Prepare data for trading strategy by calculating all necessary indicators
    """
    processed_df = df.copy()
    
    # Calculate Heikin-Ashi candles
    processed_df = calculate_heikin_ashi(processed_df)
    
    # Calculate technical indicators
    processed_df['ATR'] = calculate_atr(processed_df, atr_period)
    
    # Calculate ADX indicators (short term and long term)
    _, _, adx_short = calculate_adx(processed_df, adx_short_period)
    _, _, adx_long = calculate_adx(processed_df, adx_long_period)
    processed_df['adx'] = adx_short
    processed_df['adxl'] = adx_long
    
    # Calculate returns
    processed_df['DailyReturns'] = processed_df['close'].pct_change()
    processed_df['WeeklyReturns'] = processed_df['DailyReturns'].rolling(5).sum()
    processed_df['CumulativeReturns'] = (1 + processed_df['DailyReturns']).cumprod()
    
    # Initialize position and signal columns
    processed_df['Position'] = 0
    processed_df['signals'] = 0
    
    return processed_df

def heikin_ashi_trading_strategy(df, atr_multiplier=3):
    """
    Implement the Heikin-Ashi trading strategy with ATR-based stop losses
    
    Parameters:
    - df: DataFrame with price data and indicators
    - atr_multiplier: Multiplier for ATR to set stop loss distance
    
    Returns:
    - DataFrame with trading signals
    """
    result_df = df.copy()
    stop_loss = 0
    
    for i in range(1, len(result_df)-1):
        # Get the previous and current values of the indicators and candles
        prev_ha_close = result_df.loc[i-1, 'HA_Close']
        curr_ha_close = result_df.loc[i, 'HA_Close']
        prev_ha_open = result_df.loc[i-1, 'HA_Open']
        curr_ha_open = result_df.loc[i, 'HA_Open']
        curr_atr = result_df.loc[i, 'ATR']

        # Long entry condition
        if (prev_ha_close > prev_ha_open and 
            curr_ha_close > curr_ha_open and 
            result_df['adx'][i] < 25 and 
            result_df['adxl'][i] < 50 and 
            result_df['Position'][i] != 1):
            
            # Enter a long position and set the signal
            result_df.loc[i+1, 'Position'] = 1
            result_df.loc[i, 'signals'] = result_df.loc[i+1, 'Position'] - result_df.loc[i, 'Position']
            # Set the initial stop loss
            stop_loss = result_df.loc[i, 'high'] - curr_atr * (atr_multiplier+1)

        # Short entry condition
        elif (prev_ha_close > prev_ha_open and 
              curr_ha_close < curr_ha_open and 
              result_df['adx'][i] < 25 and 
              result_df['adxl'][i] < 50 and 
              result_df['Position'][i] != -1):
            
            # Enter a short position and set the signal
            result_df.loc[i+1, 'Position'] = -1
            result_df.loc[i, 'signals'] = result_df.loc[i+1, 'Position'] - result_df.loc[i, 'Position']
            # Set the initial stop loss
            stop_loss = result_df.loc[i, 'low'] + curr_atr * (atr_multiplier-1)

        # Secondary long condition (high ADX with negative weekly returns)
        elif (result_df['adx'][i] > 60 and 
              result_df['Position'][i] != 1 and 
              result_df['WeeklyReturns'][i] < 0):
            
            result_df.loc[i+1, 'Position'] = 1
            result_df.loc[i, 'signals'] = result_df.loc[i+1, 'Position'] - result_df.loc[i, 'Position']
            stop_loss = result_df.loc[i, 'high'] - curr_atr * (atr_multiplier+1)

        # Secondary short condition (high ADX with positive weekly returns)
        elif (result_df['adx'][i] > 60 and 
              result_df['Position'][i] != -1 and 
              result_df['WeeklyReturns'][i] > 0):
            
            result_df.loc[i+1, 'Position'] = -1
            result_df.loc[i, 'signals'] = result_df.loc[i+1, 'Position'] - result_df.loc[i, 'Position']
            stop_loss = result_df.loc[i, 'low'] + curr_atr * (atr_multiplier-1)

        # Position and stop loss management
        else:
            # Long position management
            if result_df.loc[i, 'Position'] == 1:
                # Carry over the position
                result_df.loc[i+1, 'Position'] = 1
                # Check if stop loss is hit
                if result_df.loc[i, 'low'] < stop_loss:
                    # Exit the position
                    result_df.loc[i+1, 'Position'] = 0
                    result_df.loc[i, 'signals'] = -1
                else:
                    # Update the trailing stop loss
                    stop_loss = max(stop_loss, result_df.loc[i, 'high'] - curr_atr * (atr_multiplier+1))
            
            # Short position management
            elif result_df.loc[i, 'Position'] == -1:
                # Carry over the position
                result_df.loc[i+1, 'Position'] = -1
                # Check if stop loss is hit
                if result_df.loc[i, 'high'] > stop_loss:
                    # Exit the position
                    result_df.loc[i+1, 'Position'] = 0
                    result_df.loc[i, 'signals'] = 1
                else:
                    # Update the trailing stop loss
                    stop_loss = min(stop_loss, result_df.loc[i, 'low'] + curr_atr * (atr_multiplier-1))
            
            # No position
            else:
                # Carry over the neutral position
                result_df.loc[i+1, 'Position'] = 0
    
    return result_df

def calculate_performance_metrics(df):
    """
    Calculate strategy performance metrics
    
    Parameters:
    - df: DataFrame with position and signals
    
    Returns:
    - DataFrame with performance metrics and values for trades, returns, etc.
    """
    result_df = df.copy()
    
    # Calculate strategy returns
    result_df['StrategyReturns'] = result_df['Position'] * result_df['DailyReturns']
    
    # Count trades and apply transaction costs
    trades = 0
    for i in range(1, len(result_df)):
        if result_df.loc[i, 'signals'] != 0:
            trades += abs(result_df.loc[i, 'signals'])
            # Apply transaction cost (0.05%)
            result_df.loc[i, 'StrategyReturns'] -= 0.0005 * abs(result_df.loc[i, 'signals'])
    
    # Calculate cumulative returns
    result_df['CumulativeStrategyReturns'] = (1 + result_df['StrategyReturns']).cumprod()
    result_df['StrategyReturns'].iloc[0] = 1
    
    # Calculate maximum drawdown
    result_df['MDD'] = ((result_df['StrategyReturns'].cumsum().cummax() - 
                         result_df['StrategyReturns'].cumsum()) / 
                         result_df['StrategyReturns'].cumsum().cummax()) * 100
    
    # Calculate Sharpe ratio
    avg_return = result_df['StrategyReturns'].mean()
    std_dev = result_df['StrategyReturns'].std()
    sharpe_ratio = (365 * avg_return) / (np.sqrt(365) * std_dev)
    
    # Maximum drawdown
    max_drawdown = result_df['MDD'].max()
    
    performance_metrics = {
        'trades': trades,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown,
        'avg_return': avg_return,
        'std_dev': std_dev
    }
    
    return result_df, performance_metrics

def plot_strategy_results(df, performance_metrics):
    """
    Visualize the strategy performance with charts
    
    Parameters:
    - df: DataFrame with strategy results
    - performance_metrics: Dictionary with performance metrics
    """
    plt.figure(figsize=(12, 10))
    
    # Plot price and strategy returns
    plt.subplot(3, 1, 1)
    plt.plot(df['CumulativeReturns'], label='Buy and Hold')
    plt.plot(df['StrategyReturns'].cumsum(), label='HA+2ADX : static')
    plt.legend()
    plt.title('Cumulative Returns')
    
    # Plot strategy returns comparison
    plt.subplot(3, 1, 3)
    plt.plot(df['CumulativeStrategyReturns'], label='HA+2ADX: compound')
    plt.plot(df['StrategyReturns'].cumsum(), label='HA+2ADX: static')
    plt.legend()
    plt.title('Strategy Returns')
    
    # Plot drawdown
    plt.subplot(3, 1, 2)
    plt.plot(df['MDD'], label='Drawdown')
    plt.legend()
    plt.title('Drawdown Analysis')
    
    plt.tight_layout()
    plt.show()
    
    # Print performance metrics
    print(f"Sharpe Ratio: {performance_metrics['sharpe_ratio']:.4f}")
    print(f"No of trades: {performance_metrics['trades']:.0f}")
    print(f"Max Drawdown: {performance_metrics['max_drawdown']:.4f} %")

def run_strategy_pipeline(symbol, start_date, end_date, atr_period=10, atr_multiplier=3):
    """
    Complete end-to-end pipeline to fetch data, run the strategy, analyze performance,
    and visualize results.
    
    Parameters:
    - symbol: Trading pair or symbol to analyze
    - start_date: Start date for historical data
    - end_date: End date for historical data
    - atr_period: Period for ATR calculation
    - atr_multiplier: Multiplier for ATR in stop loss calculation
    
    Returns:
    - DataFrame with complete strategy results
    - Dictionary with performance metrics
    """
    # Step 1: Fetch historical data
    df = fetch_data(symbol, start_date, end_date)
    if df is None:
        print("Failed to fetch data. Exiting.")
        return None, None
    
    # Step 2: Prepare data with indicators
    df = prepare_data(df, atr_period)
    
    # Step 3: Run the Heikin-Ashi trading strategy
    df = heikin_ashi_trading_strategy(df, atr_multiplier)
    
    # Step 4: Calculate performance metrics
    df, performance_metrics = calculate_performance_metrics(df)
    
    # Step 5: Visualize results
    plot_strategy_results(df, performance_metrics)
    
    # Step 6: Save results to CSV (optional)
    df.to_csv(f"{symbol.replace('-', '_')}_strategy_results.csv")
    
    return df, performance_metrics

# Example usage
if __name__ == "__main__":
    # Define parameters
    symbol = "BTC-USD"
    start_date = datetime(2022, 1, 13)
    end_date = datetime(2023, 12, 12)
    atr_period = 10
    atr_multiplier = 3
    
    # Run the strategy pipeline
    results_df, metrics = run_strategy_pipeline(
        symbol, start_date, end_date, atr_period, atr_multiplier
    )