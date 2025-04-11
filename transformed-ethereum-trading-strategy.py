import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pykalman import KalmanFilter
from hurst import compute_Hc
from binance.client import Client
import warnings 
warnings.filterwarnings("ignore")

# Functions for Ethereum-specific indicators
def calculate_rsi(prices, window=14):
    """
    Calculate RSI for the given price series.
    """
    delta = prices.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=window).mean()
    avg_loss = pd.Series(loss).rolling(window=window).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def calculate_atr(high, low, close, window=12):
    """
    Calculate ATR for the given price series.
    """
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    true_range = pd.DataFrame({'high_low': high_low, 'high_close': high_close, 'low_close': low_close}).max(axis=1)
    return true_range.rolling(window=window).mean()

def apply_kalman_filter(prices, transition_covariance=0.01):
    """
    Apply Kalman filter to the price series.
    """
    prices_array = prices.values.reshape(-1, 1)
    kf = KalmanFilter(
        transition_matrices=[1],
        observation_matrices=[1],
        initial_state_mean=prices_array[0],
        initial_state_covariance=1,
        observation_covariance=0.1,
        transition_covariance=transition_covariance
    )
    state_means, _ = kf.filter(prices_array)
    return pd.Series(state_means.flatten(), index=prices.index)

def calculate_correlation(series1, series2, window=7):
    """
    Calculate the correlation between two series with a rolling window.
    """
    return series1.rolling(window).corr(series2)

def calculate_hurst_exponent(ts, window=5*24):
    """
    Calculate the Hurst exponent using rescaled range analysis.
    """
    def hurst(ts):
        H, _, _ = compute_Hc(ts)
        return H
    
    return ts.rolling(window).apply(hurst, raw=True)

def calculate_bollinger_bands(prices, window=20, num_std=2):
    """
    Calculate Bollinger Bands for the given price series.
    """
    middle = prices.rolling(window=window).mean()
    std = prices.rolling(window=window).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return middle, upper, lower

def calculate_supertrend(high, low, close, atr, period=12, multiplier=2.5):
    """
    Calculate Supertrend indicator.
    """
    middle_band = (high + low) / 2
    upper_band = middle_band + (multiplier * atr)
    lower_band = middle_band - (multiplier * atr)
    
    supertrend = np.zeros(len(close))
    supertrend_direction = np.zeros(len(close))
    
    # First value initialization
    supertrend[0] = upper_band.iloc[0]
    trend = 1
    
    for i in range(1, len(close)):
        if close.iloc[i] > supertrend[i-1]:
            trend = 1
        elif close.iloc[i] < supertrend[i-1]:
            trend = -1
        
        if trend == 1:
            if lower_band.iloc[i] < supertrend[i-1]:
                supertrend[i] = supertrend[i-1]
            else:
                supertrend[i] = lower_band.iloc[i]
                
        elif trend == -1:
            if upper_band.iloc[i] > supertrend[i-1]:
                supertrend[i] = supertrend[i-1]
            else:
                supertrend[i] = upper_band.iloc[i]
        
        supertrend_direction[i] = trend
    
    return pd.Series(supertrend, index=close.index), pd.Series(supertrend_direction, index=close.index)

def identify_regimes(price, filtered_price, window=5, delta=0.8, h_factor=1.5):
    """
    Identify price regimes based on CUSUM analysis.
    """
    rolling_sigma = price.rolling(window=window).std()
    k = delta * rolling_sigma
    
    # Calculate CUSUM
    n = len(price)
    S_hi = np.zeros(n)
    S_lo = np.zeros(n)
    price_array = price.values
    filtered_array = filtered_price.values
    
    for i in range(1, n):
        S_hi[i] = max(0, S_hi[i-1] + (price_array[i] - filtered_array[i] - k.iloc[i]))
        S_lo[i] = max(0, S_lo[i-1] + (-price_array[i] + filtered_array[i] - k.iloc[i]))
    
    # Define regime thresholds
    rolling_h = h_factor * rolling_sigma
    
    # Identify bullish regimes
    regime = pd.Series('bearish', index=price.index)
    regime[pd.Series(S_hi, index=price.index) > rolling_h] = 'bullish'
    
    return regime

def prepare_data(btc_csv_path=None, eth_csv_path=None, use_api=False, symbol="ETHUSDT", interval="1h", limit=5000):
    """
    Prepare complete dataset for the trading strategy by:
    1. Loading data from CSV or fetching from Binance API
    2. Calculating technical indicators
    
    Parameters:
    - btc_csv_path: Path to BTC CSV file
    - eth_csv_path: Path to ETH CSV file
    - use_api: Whether to fetch data from Binance API
    - symbol: Trading pair to analyze if using API
    - interval: Timeframe for the data if using API
    - limit: Number of candles to fetch if using API
    
    Returns:
    - Complete DataFrame ready for the strategy
    """
    if not use_api and btc_csv_path and eth_csv_path and os.path.exists(btc_csv_path) and os.path.exists(eth_csv_path):
        print(f"Loading data from {btc_csv_path} and {eth_csv_path}...")
        
        # Load BTC and ETH data from the specified CSV paths
        btc_data = pd.read_csv(btc_csv_path, parse_dates=['datetime'])
        eth_data = pd.read_csv(eth_csv_path, parse_dates=['datetime'])
        
        # Rename columns for consistency and merge datasets on 'datetime'
        btc_data = btc_data[['datetime', 'open', 'high', 'low', 'close', 'volume']].rename(
            columns={'open': 'btc_open', 'high': 'btc_high', 'low': 'btc_low', 'close': 'btc_close', 'volume': 'btc_volume'}
        )
        eth_data = eth_data[['datetime', 'open', 'high', 'low', 'close', 'volume']].rename(
            columns={'open': 'eth_open', 'high': 'eth_high', 'low': 'eth_low', 'close': 'eth_close', 'volume': 'eth_volume'}
        )
        
        # Merge datasets on datetime
        df = pd.merge(btc_data, eth_data, on='datetime', how='inner')
        df = df.set_index('datetime')
        
    else:
        print(f"Fetching {symbol} data from Binance ({interval} interval)...")
        client = Client()
        
        # Get ETH data
        eth_klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        eth_df = pd.DataFrame(eth_klines, columns=[
            'timestamp', 'eth_open', 'eth_high', 'eth_low', 'eth_close', 'eth_volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Get BTC data for correlation
        btc_klines = client.get_klines(symbol="BTCUSDT", interval=interval, limit=limit)
        btc_df = pd.DataFrame(btc_klines, columns=[
            'timestamp', 'btc_open', 'btc_high', 'btc_low', 'btc_close', 'btc_volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert timestamp to datetime
        eth_df['timestamp'] = pd.to_datetime(eth_df['timestamp'], unit='ms')
        btc_df['timestamp'] = pd.to_datetime(btc_df['timestamp'], unit='ms')
        
        # Convert numeric columns to float
        for col in ['eth_open', 'eth_high', 'eth_low', 'eth_close', 'eth_volume']:
            eth_df[col] = eth_df[col].astype(float)
        for col in ['btc_open', 'btc_high', 'btc_low', 'btc_close', 'btc_volume']:
            btc_df[col] = btc_df[col].astype(float)
        
        # Merge datasets on timestamp
        df = pd.merge(btc_df, eth_df, on='timestamp', how='inner', suffixes=('', '_eth'))
        df = df.set_index('timestamp')
    
    # Calculate RSI
    df['btc_rsi'] = calculate_rsi(df['btc_close'])
    df['eth_rsi'] = calculate_rsi(df['eth_close'])
    
    # Calculate ATR
    df['btc_atr'] = calculate_atr(df['btc_high'], df['btc_low'], df['btc_close'])
    df['eth_atr'] = calculate_atr(df['eth_high'], df['eth_low'], df['eth_close'])
    
    # Apply Kalman Filter
    df['btc_close_filtered'] = apply_kalman_filter(df['btc_close'])
    df['eth_close_filtered'] = apply_kalman_filter(df['eth_close'])
    
    # Calculate Correlation
    df['btc_eth_correlation'] = calculate_correlation(df['btc_close'], df['eth_close'])
    
    # Calculate Hurst Exponent
    df['btc_hurst'] = calculate_hurst_exponent(df['btc_close'])
    df['eth_hurst'] = calculate_hurst_exponent(df['eth_close'])
    
    # Calculate Bollinger Bands
    df['btc_bollinger_middle'], df['btc_bollinger_upper'], df['btc_bollinger_lower'] = calculate_bollinger_bands(df['btc_close'])
    df['eth_bollinger_middle'], df['eth_bollinger_upper'], df['eth_bollinger_lower'] = calculate_bollinger_bands(df['eth_close'])
    
    # Calculate Supertrend
    df['btc_supertrend'], df['btc_supertrend_direction'] = calculate_supertrend(
        df['btc_high'], df['btc_low'], df['btc_close'], df['btc_atr']
    )
    df['eth_supertrend'], df['eth_supertrend_direction'] = calculate_supertrend(
        df['eth_high'], df['eth_low'], df['eth_close'], df['eth_atr']
    )
    
    # Identify Regimes
    df['btc_regime'] = identify_regimes(df['btc_close'], df['btc_close_filtered'])
    df['eth_regime'] = identify_regimes(df['eth_close'], df['eth_close_filtered'])
    
    # Add volatility and momentum metrics
    df['eth_volatility'] = df['eth_high'] - df['eth_low']
    df['eth_momentum'] = df['eth_close'] - df['eth_close'].shift(14)
    df['eth_volume_change'] = df['eth_volume'].pct_change().fillna(0)
    df['eth_price_change'] = df['eth_close'].pct_change().fillna(0)
    
    # Initialize Signal and Position columns
    df['signals'] = 0
    df['position'] = 0
    df['trade_type'] = ""
    
    # Ensure we have data within the required timeframe (2020-2023)
    if not use_api:
        df = df[(df.index >= '2020-01-01') & (df.index <= '2023-12-31')].copy()
    
    return df

def execute_trade_strategy(df):
    """
    Implement the ethereum trading strategy.
    
    Parameters:
    - df: DataFrame with price data and technical indicators
    
    Returns:
    - DataFrame with trading signals and positions
    """
    # Strategy Thresholds and Parameters
    CORRELATION_THRESHOLD = 0.6 
    HURST_THRESHOLD = 0.5
    RSI_THRESHOLD_HIGH = 70
    RSI_THRESHOLD_LOW = 30
    BTC_ATR_THRESHOLD_STOP = 0.025  # Above which position will be exited
    BTC_ATR_THRESHOLD_TRADE = 0.01  # Below which trading can take place

    TRAILING_STOPLOSS_PCT = 0.10    # Trailing stoploss percentage
    MAX_HOLDING_PERIOD = 28 * 24    # 4 weeks in hours 
    COOLDOWN_PERIOD = 24            # Cooldown period after stop-loss hit
    
    # Initialize Trading Variables
    current_position = 0    # 0 = no position, 1 = long position, -1 = short position
    entry_price = None
    entry_date = None
    highest_since_entry = None
    lowest_since_entry = None
    last_trailing_stop_time = None
    
    # Main loop to process data row by row
    for i in range(1, len(df)):
        current_price = df['eth_close'].iloc[i]
        current_time = df.index[i]
        high_price = df['eth_high'].iloc[i]
        low_price = df['eth_low'].iloc[i]

        # Halt trading during cooldown period
        if last_trailing_stop_time is not None:
            time_since_trailing_stop = (current_time - last_trailing_stop_time).total_seconds() / 3600
            if time_since_trailing_stop < COOLDOWN_PERIOD:
                continue
        
        # Close position at end of data
        if i == len(df) - 1 and current_position != 0:
            if current_position == 1:
                df.iloc[i, df.columns.get_loc('signals')] = -1
            else:
                df.iloc[i, df.columns.get_loc('signals')] = 1
            df.iloc[i, df.columns.get_loc('trade_type')] = 'close'
            current_position = 0
            continue
        
        # Handle trailing stop-loss for long positions
        if current_position == 1:
            highest_since_entry = max(highest_since_entry or current_price, current_price)
            trailing_stop_price = highest_since_entry * (1 - TRAILING_STOPLOSS_PCT)
            lowest_24hr = df['eth_low'].iloc[max(0, i-24):i+1].min()
            stop_price = (lowest_24hr + trailing_stop_price) / 2 if lowest_24hr < trailing_stop_price else lowest_24hr
                    
            if current_price <= stop_price:
                df.iloc[i, df.columns.get_loc('signals')] = -1
                df.iloc[i, df.columns.get_loc('trade_type')] = 'close'
                current_position = 0
                last_trailing_stop_time = current_time
                entry_price = None
                entry_date = None
                highest_since_entry = None
                continue
                    
        # Handle trailing stop-loss for short positions  
        elif current_position == -1:
            lowest_since_entry = min(lowest_since_entry or low_price, low_price)
            trailing_stop_price = lowest_since_entry * (1 + TRAILING_STOPLOSS_PCT)

            if current_price >= trailing_stop_price:
                df.iloc[i, df.columns.get_loc('signals')] = 1
                df.iloc[i, df.columns.get_loc('trade_type')] = 'close'
                current_position = 0
                last_trailing_stop_time = current_time
                entry_price = None
                entry_date = None
                lowest_since_entry = None
                continue
        
        # Check max holding period and volatility-based stop-loss
        if current_position != 0 and entry_date is not None:
            time_since_entry = (current_time - entry_date).total_seconds() / 3600
            if time_since_entry >= MAX_HOLDING_PERIOD:  # Exceeded max holding period
                if current_position == 1:
                    df.iloc[i, df.columns.get_loc('signals')] = -1
                else:
                    df.iloc[i, df.columns.get_loc('signals')] = 1
                df.iloc[i, df.columns.get_loc('trade_type')] = 'close'
                current_position = 0
                last_trailing_stop_time = current_time
                entry_price = None
                entry_date = None
                highest_since_entry = None
                lowest_since_entry = None
                continue
                
            if df['btc_atr'].iloc[i] > BTC_ATR_THRESHOLD_STOP * df['btc_open'].iloc[i]:  # BTC indicating high-volatility
                if current_position == 1:
                    df.iloc[i, df.columns.get_loc('signals')] = -1
                else:
                    df.iloc[i, df.columns.get_loc('signals')] = 1
                df.iloc[i, df.columns.get_loc('trade_type')] = 'close'
                current_position = 0
                last_trailing_stop_time = current_time
                entry_price = None
                entry_date = None
                highest_since_entry = None
                lowest_since_entry = None
                continue

        # Trading logic under low volatility and high correlation conditions
        if (df['btc_atr'].iloc[i] < BTC_ATR_THRESHOLD_TRADE * df['btc_open'].iloc[i] and 
            df['btc_eth_correlation'].iloc[i] > CORRELATION_THRESHOLD and 
            df['eth_hurst'].iloc[i] > HURST_THRESHOLD):  

            # LONG ENTRY CONDITIONS
            if current_position == 0 and (
                df['btc_rsi'].iloc[i] > RSI_THRESHOLD_HIGH and 
                df['btc_regime'].iloc[i] == 'bullish' and
                df['btc_close'].iloc[i] > df['btc_bollinger_middle'].iloc[i] and
                df['eth_supertrend_direction'].iloc[i] == 1):

                df.iloc[i, df.columns.get_loc('signals')] = 1
                df.iloc[i, df.columns.get_loc('trade_type')] = 'long'
                df.iloc[i, df.columns.get_loc('position')] = 1
                current_position = 1
                entry_price = current_price
                entry_date = current_time
                highest_since_entry = current_price
                
            # SHORT ENTRY CONDITIONS
            elif current_position == 0 and (
                df['btc_rsi'].iloc[i] < RSI_THRESHOLD_LOW and
                df['btc_regime'].iloc[i] == 'bearish' and
                df['btc_close'].iloc[i] < df['btc_bollinger_lower'].iloc[i] and
                df['eth_supertrend_direction'].iloc[i] == -1):  

                df.iloc[i, df.columns.get_loc('signals')] = -1
                df.iloc[i, df.columns.get_loc('trade_type')] = 'short'
                df.iloc[i, df.columns.get_loc('position')] = -1
                current_position = -1
                entry_price = current_price
                entry_date = current_time
                lowest_since_entry = current_price
                              
            # LONG EXIT CONDITIONS
            elif current_position == 1 and (   
                df['btc_rsi'].iloc[i] < RSI_THRESHOLD_LOW and
                df['eth_rsi'].iloc[i] < df['eth_rsi'].iloc[i-1] and
                df['btc_regime'].iloc[i] == 'bearish' and
                df['btc_close'].iloc[i] < df['btc_bollinger_lower'].iloc[i] and
                df['eth_supertrend_direction'].iloc[i] == -1):  
                            
                df.iloc[i, df.columns.get_loc('signals')] = -1
                df.iloc[i, df.columns.get_loc('trade_type')] = 'close'
                df.iloc[i, df.columns.get_loc('position')] = 0
                current_position = 0
                entry_price = None
                entry_date = None
                highest_since_entry = None

            # SHORT EXIT CONDITIONS
            elif current_position == -1 and (
                df['btc_rsi'].iloc[i] > RSI_THRESHOLD_HIGH and
                df['btc_rsi'].iloc[i-1] > RSI_THRESHOLD_HIGH and
                df['eth_rsi'].iloc[i] > df['eth_rsi'].iloc[i-1] and
                df['btc_regime'].iloc[i] == 'bullish' and
                df['btc_close'].iloc[i] > df['btc_bollinger_middle'].iloc[i] and
                df['eth_supertrend_direction'].iloc[i] == 1):
                
                df.iloc[i, df.columns.get_loc('signals')] = 1
                df.iloc[i, df.columns.get_loc('trade_type')] = 'close'
                df.iloc[i, df.columns.get_loc('position')] = 0
                current_position = 0
                entry_price = None
                entry_date = None
                lowest_since_entry = None
        
        # Update position column for continuous tracking
        if current_position != 0:
            df.iloc[i, df.columns.get_loc('position')] = current_position
    
    return df

def analyze_strategy_performance(df, initial_balance=10000):
    """
    Calculate performance metrics for the trading strategy.
    
    Parameters:
    - df: DataFrame with position and price columns
    - initial_balance: Initial balance for the simulation
    
    Returns:
    - Dictionary of performance metrics
    """
    # Calculate strategy returns
    df['trade_returns'] = df['position'].shift(1) * df['eth_close'].pct_change()
    df['strategy_returns'] = (1 + df['trade_returns'].fillna(0)).cumprod() - 1
    
    # Calculate Buy & Hold returns for comparison
    df['buy_hold_returns'] = df['eth_close'].pct_change()
    df['buy_hold_strategy'] = (1 + df['buy_hold_returns'].fillna(0)).cumprod() - 1
    
    # Calculate performance metrics
    total_return = df['strategy_returns'].iloc[-1]
    buy_hold_return = df['buy_hold_strategy'].iloc[-1]
    
    # Calculate annualized metrics
    days = (df.index[-1] - df.index[0]).days
    annual_factor = 365 / days if days > 0 else 1
    annualized_return = ((1 + total_return) ** annual_factor) - 1
    
    # Calculate volatility and Sharpe ratio
    daily_returns = df['trade_returns'].fillna(0)
    volatility = daily_returns.std() * np.sqrt(365)  # Annualized
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    # Maximum drawdown
    cumulative_returns = df['strategy_returns']
    peak = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - peak) / (peak + 1)  # Add 1 to avoid division by zero
    max_drawdown = drawdown.min()
    
    # Trade statistics
    trades = df[df['signals'] != 0]
    total_trades = len(trades)
    
    # Calculate win/loss ratio
    winning_trades = 0
    losing_trades = 0
    
    for i in range(1, len(trades)):
        if trades.iloc[i]['signals'] == 1 and trades.iloc[i-1]['signals'] == -1:  # Exit short
            if trades.iloc[i-1]['eth_close'] > trades.iloc[i]['eth_close']:
                winning_trades += 1
            else:
                losing_trades += 1
        elif trades.iloc[i]['signals'] == -1 and trades.iloc[i-1]['signals'] == 1:  # Exit long
            if trades.iloc[i]['eth_close'] > trades.iloc[i-1]['eth_close']:
                winning_trades += 1
            else:
                losing_trades += 1
    
    win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
    
    return {
        'total_return': total_return * 100,  # Convert to percentage
        'buy_hold_return': buy_hold_return * 100,
        'annualized_return': annualized_return * 100,
        'volatility': volatility * 100,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown * 100,
        'win_rate': win_rate,
        'total_trades': total_trades,
    }

def print_performance_metrics(metrics):
    """Print formatted performance metrics"""
    print("\n===== ETHEREUM TRADING STRATEGY PERFORMANCE =====")
    print(f"Total Return: {metrics['total_return']:.2f}%")
    print(f"Buy & Hold Return: {metrics['buy_hold_return']:.2f}%")
    print(f"Annualized Return: {metrics['annualized_return']:.2f}%")
    print(f"Volatility: {metrics['volatility']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    print(f"Max Drawdown: -{metrics['max_drawdown']:.2f}%")
    print(f"Win Rate: {metrics['win_rate']:.2f}%")
    print(f"Total Trades: {metrics['total_trades']}")

def plot_strategy_results(df, title="Ethereum Trading Strategy Results"):
    """
    Plot the trading strategy results including:
    - Price chart with buy/sell signals
    - Equity curve
    - RSI and key indicators
    
    Parameters:
    - df: DataFrame with strategy results
    - title: Title for the plot
    """
    # Create plot with subplots
    fig, axs = plt.subplots(4, 1, figsize=(16, 20), gridspec_kw={'height_ratios': [2, 1, 1, 1]})
    
    # Plot 1: Price chart with buy/sell markers
    axs[0].plot(df.index, df['eth_close'], label='ETH Price', color='gray')
    
    # Add Buy markers (long entries)
    buy_signals = df[(df['signals'] == 1) & (df['trade_type'] == 'long')]
    if not buy_signals.empty:
        axs[0].scatter(buy_signals.index, buy_signals['eth_close'], marker='^', color='green', s=100, label='Long Entry')
    
    # Add Sell markers (short entries)
    sell_signals = df[(df['signals'] == -1) & (df['trade_type'] == 'short')]
    if not sell_signals.empty:
        axs[0].scatter(sell_signals.index, sell_signals['eth_close'], marker='v', color='red', s=100, label='Short Entry')
    
    # Add Close markers
    close_signals = df[df['trade_type'] == 'close']
    if not close_signals.empty:
        axs[0].scatter(close_signals.index, close_signals['eth_close'], marker='X', color='blue', s=80, label='Position Close')
    
    # Add Bollinger Bands
    axs[0].plot(df.index, df['eth_bollinger_upper'], 'r--', alpha=0.3, label='Upper BB')
    axs[0].plot(df.index, df['eth_bollinger_middle'], 'b--', alpha=0.3, label='Middle BB')
    axs[0].plot(df.index, df['eth_bollinger_lower'], 'g--', alpha=0.3, label='Lower BB')
    
    axs[0].set_title(f'{title} - ETH Price with Trading Signals')
    axs[0].set_ylabel('Price (USD)')
    axs[0].legend(loc='upper left')
    axs[0].grid(True)
    
    # Plot 2: Strategy returns vs. Buy & Hold
    axs[1].plot(df.index, df['strategy_returns'] * 100, label='Strategy Returns', color='blue')
    axs[1].plot(df.index, df['buy_hold_strategy'] * 100, label='Buy & Hold', color='gray', alpha=0.5)
    axs[1].set_title('Cumulative Returns (%)')
    axs[1].set_ylabel('Returns (%)')
    axs[1].legend()
    axs[1].grid(True)
    
    # Plot 3: RSI for BTC and ETH
    axs[2].plot(df.index, df['btc_rsi'], label='BTC RSI', color='orange')
    axs[2].plot(df.index, df['eth_rsi'], label='ETH RSI', color='purple')
    axs[2].axhline(y=70, color='r', linestyle='--', label='Overbought (70)')
    axs[2].axhline(y=30, color='g', linestyle='--', label='Oversold (