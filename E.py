import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os.path
from binance.client import Client
from ta.momentum import RSIIndicator
from ta.trend import MACD, ADXIndicator
from ta.volatility import AverageTrueRange

# Ichimoku Cloud functions
def calculate_tenkan_sen(high, low, period=9):
    """Calculate Tenkan-sen (Conversion Line)"""
    highs = high.rolling(window=period).max()
    lows = low.rolling(window=period).min()
    return (highs + lows) / 2

def calculate_kijun_sen(high, low, period=26):
    """Calculate Kijun-sen (Base Line)"""
    highs = high.rolling(window=period).max()
    lows = low.rolling(window=period).min()
    return (highs + lows) / 2

def ichimoku_cloud(high, low, close, tenkan_period=9, kijun_period=26):
    """Calculate Ichimoku Cloud components"""
    tenkan_sen = calculate_tenkan_sen(high, low, tenkan_period)
    kijun_sen = calculate_kijun_sen(high, low, kijun_period)
    
    # Calculate Senkou Span A (Leading Span A)
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(kijun_period)
    
    # Calculate Senkou Span B (Leading Span B)
    senkou_span_b = ((high.rolling(window=52).max() + low.rolling(window=52).min()) / 2).shift(kijun_period)
    
    # Calculate Chikou Span (Lagging Span)
    chikou_span = close.shift(-kijun_period)
    
    return tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span

def calculate_heikin_ashi(df):
    """
    Calculate Heikin-Ashi candles from regular OHLC data
    """
    ha_df = df.copy()
    
    # First create all necessary columns
    ha_df['HA_Close'] = (df['open'] + df['high'] + df['low'] + df['close']) / 4
    ha_df['HA_Open'] = df['open'].copy()  # Initialize with regular open values
    
    # For the first candle, HA_Open equals the regular open (already set above)
    # Calculate HA_Open for the rest
    for i in range(1, len(df)):
        ha_df.iloc[i, ha_df.columns.get_loc('HA_Open')] = (ha_df['HA_Open'].iloc[i-1] + ha_df['HA_Close'].iloc[i-1]) / 2
    
    ha_df['HA_High'] = ha_df[['high', 'HA_Open', 'HA_Close']].max(axis=1)
    ha_df['HA_Low'] = ha_df[['low', 'HA_Open', 'HA_Close']].min(axis=1)
    
    return ha_df

def prepare_data(symbol="BTCUSDT", interval="1h", limit=500, use_csv=True, csv_path='1hr.csv'):
    """
    Prepare complete dataset for the trading strategy by:
    1. Loading data from CSV or fetching from Binance API
    2. Calculating technical indicators
    3. Creating Heikin-Ashi candles
    4. Processing clustering data if available
    
    Parameters:
    - symbol: Trading pair to analyze
    - interval: Timeframe for the data
    - limit: Number of candles to fetch
    - use_csv: Whether to use CSV file or fetch from API
    - csv_path: Path to the CSV file with data
    
    Returns:
    - Complete DataFrame ready for the strategy
    """
    if use_csv and os.path.exists(csv_path):
        print(f"Loading data from {csv_path}...")
        df = pd.read_csv(csv_path)
        
        # Check if all required columns are present
        required_columns = ['timestamp', 'open', 'high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            print("Falling back to Binance API...")
            use_csv = False
    elif use_csv:
        print(f"CSV file {csv_path} not found, falling back to Binance API...")
        use_csv = False
    
    if not use_csv:
        # Fetch data from Binance API
        print(f"Fetching {symbol} data from Binance ({interval} interval)...")
        client = Client()
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        
        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Convert types
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
    
    # Make sure timestamp is datetime
    if 'timestamp' in df.columns and not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Calculate technical indicators
    # RSI
    if 'RSI' not in df.columns:
        df['RSI'] = RSIIndicator(close=df['close'], window=14).rsi()
    
    # MACD
    if 'MACD' not in df.columns:
        macd = MACD(close=df['close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
    
    # EMA calculations
    df['EMA_9'] = df['close'].ewm(span=9, adjust=False).mean()
    df['EMA_14'] = df['close'].ewm(span=14, adjust=False).mean()
    
    # ATR calculation
    atr_indicator = AverageTrueRange(high=df['high'], low=df['low'], close=df['close'], window=11)
    df['ATR'] = atr_indicator.average_true_range()
    
    # ADX calculation for trend strength
    adx_indicator = ADXIndicator(high=df['high'], low=df['low'], close=df['close'], window=14)
    df['ADX'] = adx_indicator.adx()
    
    # Ichimoku Cloud components
    tenkan_sen, kijun_sen, senkou_span_a, senkou_span_b, chikou_span = ichimoku_cloud(
        df['high'], df['low'], df['close'], 15, 30
    )
    df['Tenkan_Sen'] = tenkan_sen
    df['Kijun_Sen'] = kijun_sen
    df['Senkou_Span_A'] = senkou_span_a
    df['Senkou_Span_B'] = senkou_span_b
    df['Chikou_Span'] = chikou_span
    
    # Calculate Heikin-Ashi candles
    df = calculate_heikin_ashi(df)
    
    # Add other features from original code
    df['HL_Volatility'] = df['high'] - df['low']
    df['Body'] = df['close'] - df['open']
    df['Volume_Change'] = df['volume'].pct_change().fillna(0)
    df['Price_Change'] = df['close'].diff().fillna(0)
    df['Pct_Change'] = df['close'].pct_change().fillna(0)
    df['Upper_Shadow'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['Lower_Shadow'] = df[['open', 'close']].min(axis=1) - df['low']
    df['SMA_14'] = df['close'].rolling(window=14).mean()
    df['Momentum'] = df['close'] - df['close'].shift(14)
    df['Rolling_Volatility_14'] = df['close'].pct_change().rolling(14).std()
    
    # Initialize Position and Signal columns for the strategy
    df['Position'] = 0
    df['Signal'] = 0
    
    # Reset index to make it easier to work with iloc in the strategy
    if 'timestamp' in df.columns:
        df = df.set_index('timestamp')
    
    return df

def execute_trade_strategy(df):
    """
    Implement the original trading strategy but structured like in A.py
    
    Parameters:
    - df: DataFrame with price data and technical indicators
    
    Returns:
    - DataFrame with trading signals and positions
    """
    # Make a copy of the dataframe
    df_copy = df.copy()
    
    # Reset index for easier iteration
    if df_copy.index.name == 'timestamp':
        df_copy = df_copy.reset_index()
    
    stop_loss = 0
    
    for i in range(30, len(df_copy)-1):  # Skip first 30 rows for indicator calculation
        # RSI-based signals from original strategy
        if df_copy['RSI'].iloc[i] > 85 and df_copy['Kijun_Sen'].iloc[i] > df_copy['Tenkan_Sen'].iloc[i]:
            # Consider cluster sentiment if available
            proceed_with_signal = True
            
            if ('KMeans_Sentiment' in df_copy.columns and 'DTW_Sentiment' in df_copy.columns):
                kmeans_sentiment = df_copy['KMeans_Sentiment'].iloc[i]
                dtw_sentiment = df_copy['DTW_Sentiment'].iloc[i]
                
                # Cancel sell signal if both clustering methods indicate bullish sentiment
                if kmeans_sentiment == 'Bullish' and dtw_sentiment == 'Bullish':
                    proceed_with_signal = False
            
            if proceed_with_signal and df_copy['Position'].iloc[i] != -1:
                df_copy.loc[i+1, 'Position'] = -1  # Sell/Short signal
                df_copy.loc[i, 'Signal'] = -1 - df_copy.loc[i, 'Position']
                stop_loss = df_copy['high'].iloc[i] + 2*df_copy['ATR'].iloc[i]
                
        elif df_copy['RSI'].iloc[i] < 30 and df_copy['Kijun_Sen'].iloc[i] < df_copy['Tenkan_Sen'].iloc[i]:
            # Consider cluster sentiment if available
            proceed_with_signal = True
            
            if ('KMeans_Sentiment' in df_copy.columns and 'DTW_Sentiment' in df_copy.columns):
                kmeans_sentiment = df_copy['KMeans_Sentiment'].iloc[i]
                dtw_sentiment = df_copy['DTW_Sentiment'].iloc[i]
                
                # Cancel buy signal if both clustering methods indicate bearish sentiment
                if kmeans_sentiment == 'Bearish' and dtw_sentiment == 'Bearish':
                    proceed_with_signal = False
            
            if proceed_with_signal and df_copy['Position'].iloc[i] != 1:
                df_copy.loc[i+1, 'Position'] = 1  # Buy signal
                df_copy.loc[i, 'Signal'] = 1 - df_copy.loc[i, 'Position']
                stop_loss = df_copy['low'].iloc[i] - 2*df_copy['ATR'].iloc[i]
        
        # New condition: Exit positions when sentiment changes strongly against current position
        elif ('KMeans_Sentiment' in df_copy.columns and 'DTW_Sentiment' in df_copy.columns):
            kmeans_sentiment = df_copy['KMeans_Sentiment'].iloc[i]
            dtw_sentiment = df_copy['DTW_Sentiment'].iloc[i]
            
            sentiment_score = 0
            if kmeans_sentiment == 'Bullish':
                sentiment_score += 1
            elif kmeans_sentiment == 'Bearish':
                sentiment_score -= 1
                
            if dtw_sentiment == 'Bullish':
                sentiment_score += 1
            elif dtw_sentiment == 'Bearish':
                sentiment_score -= 1
            
            # Exit long if strong bearish sentiment
            if df_copy['Position'].iloc[i] == 1 and sentiment_score < -1:
                df_copy.loc[i+1, 'Position'] = 0
                df_copy.loc[i, 'Signal'] = -1
            
            # Exit short if strong bullish sentiment
            elif df_copy['Position'].iloc[i] == -1 and sentiment_score > 1:
                df_copy.loc[i+1, 'Position'] = 0
                df_copy.loc[i, 'Signal'] = 1
        
        else:
            # Manage existing positions
            if df_copy['Position'].iloc[i] == 1:  # Currently long
                df_copy.loc[i+1, 'Position'] = 1
                # Check stop loss
                if df_copy['low'].iloc[i] < stop_loss:
                    df_copy.loc[i+1, 'Position'] = 0
                    df_copy.loc[i, 'Signal'] = -1
                else:
                    # Trail stop loss
                    stop_loss = max(stop_loss, df_copy['low'].iloc[i] - 2*df_copy['ATR'].iloc[i])
            
            elif df_copy['Position'].iloc[i] == -1:  # Currently short
                df_copy.loc[i+1, 'Position'] = -1
                # Check stop loss
                if df_copy['high'].iloc[i] > stop_loss:
                    df_copy.loc[i+1, 'Position'] = 0
                    df_copy.loc[i, 'Signal'] = 1
                else:
                    # Trail stop loss
                    stop_loss = min(stop_loss, df_copy['high'].iloc[i] + 2*df_copy['ATR'].iloc[i])
            
            else:  # No position
                df_copy.loc[i+1, 'Position'] = 0
    
    # Restore index if needed
    if 'timestamp' in df_copy.columns:
        df_copy = df_copy.set_index('timestamp')
    
    return df_copy

def analyze_strategy_performance(df, initial_balance=10000):
    """
    Calculate performance metrics for the trading strategy.
    
    Parameters:
    - df: DataFrame with Position and close price columns
    - initial_balance: Initial balance for the simulation
    
    Returns:
    - Dictionary of performance metrics
    """
    # Calculate strategy returns
    df['Strategy_Returns'] = df['Position'].shift(1) * df['close'].pct_change()
    df['Cumulative_Returns'] = (1 + df['Strategy_Returns'].fillna(0)).cumprod() - 1
    
    # Calculate Buy & Hold returns for comparison
    df['BuyHold_Returns'] = df['close'].pct_change()
    df['BuyHold_Cumulative'] = (1 + df['BuyHold_Returns'].fillna(0)).cumprod() - 1
    
    # Calculate performance metrics
    total_return = df['Cumulative_Returns'].iloc[-1]
    
    # Calculate annualized metrics
    trading_days = len(df)
    annualized_return = ((1 + total_return) ** (365 / trading_days)) - 1 if trading_days > 0 else 0
    
    # Calculate volatility and Sharpe ratio
    daily_returns = df['Strategy_Returns'].fillna(0)
    volatility = daily_returns.std() * np.sqrt(365)  # Annualized
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
    
    # Maximum drawdown
    cumulative_returns = df['Cumulative_Returns'].fillna(0)
    peak = cumulative_returns.expanding().max()
    drawdown = (cumulative_returns - peak) / (peak + 1)  # Add 1 to avoid division by zero
    max_drawdown = drawdown.min()
    
    # Trade statistics
    trades = df[df['Signal'] != 0]
    total_trades = len(trades)
    profitable_trades = len(df[df['Strategy_Returns'] > 0])
    win_rate = (profitable_trades / total_trades) * 100 if total_trades > 0 else 0
    
    return {
        'total_return': total_return * 100,  # Convert to percentage
        'annualized_return': annualized_return * 100,
        'volatility': volatility * 100,
        'sharpe_ratio': sharpe_ratio,
        'max_drawdown': max_drawdown * 100,
        'win_rate': win_rate,
        'total_trades': total_trades
    }

def print_performance_metrics(metrics):
    """Print formatted performance metrics"""
    print("\n===== STRATEGY PERFORMANCE =====")
    print(f"Total Return: {metrics['total_return']:.2f}% ")
    print(f"Annualized Return: {metrics['annualized_return']:.2f}%")
    print(f"Volatility: {metrics['volatility']:.2f}%")
    print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}  ")
    print(f"Max Drawdown: -{metrics['max_drawdown']:.2f}%")
    print(f"Win Rate: {metrics['win_rate']:.2f}%    ")
    print(f"Total Trades: {metrics['total_trades']}")

def plot_strategy_results(df, title="Strategy Performance"):
    """
    Plot the trading strategy results including:
    - Price chart with buy/sell signals
    - Equity curve
    - RSI indicator
    - Cluster sentiment if available
    
    Parameters:
    - df: DataFrame with strategy results
    - title: Title for the plot
    """
    # Create plot with subplots
    fig, axs = plt.subplots(3, 1, figsize=(16, 16), gridspec_kw={'height_ratios': [2, 1, 1]})
    
    # Plot 1: Price chart with buy/sell markers
    axs[0].plot(df.index, df['close'], label='BTC Price', color='gray')
    
    # Add Buy markers (signals of 1)
    buy_signals = df[df['Signal'] == 1]
    if not buy_signals.empty:
        axs[0].scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', s=100, label='Buy')
    
    # Add Sell markers (signals of -1)
    sell_signals = df[df['Signal'] == -1]
    if not sell_signals.empty:
        axs[0].scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', s=100, label='Sell')
    
    axs[0].set_title(f'{title} - Price with Buy/Sell Signals')
    axs[0].set_ylabel('Price (USD)')
    axs[0].legend()
    axs[0].grid(True)
    
    # Plot 2: Strategy returns vs Buy & Hold
    axs[1].plot(df.index, df['Cumulative_Returns'] * 100, label='Strategy Returns', color='blue')
    axs[1].plot(df.index, df['BuyHold_Cumulative'] * 100, label='Buy & Hold', color='gray', alpha=0.5)
    axs[1].set_title('Cumulative Returns')
    axs[1].set_ylabel('Returns (%)')
    axs[1].legend()
    axs[1].grid(True)
    
    # Plot 3: RSI with overbought/oversold lines and cluster sentiment if available
    if 'RSI' in df.columns:
        axs[2].plot(df.index, df['RSI'], label='RSI', color='purple')
        axs[2].axhline(y=85, color='r', linestyle='--', label='Overbought (85)')
        axs[2].axhline(y=30, color='g', linestyle='--', label='Oversold (30)')
        axs[2].set_title('RSI Indicator')
        axs[2].set_ylabel('RSI Value')
        axs[2].legend(loc='upper left')
        axs[2].grid(True)
        
        # Plot cluster sentiment if available on secondary y-axis
        if 'KMeans_Sentiment' in df.columns and 'DTW_Sentiment' in df.columns:
            ax2 = axs[2].twinx()
            
            # Convert sentiment to numeric for plotting
            sentiment_map = {'Bearish': -1, 'Sideways': 0, 'Bullish': 1}
            kmeans_sentiment_numeric = df['KMeans_Sentiment'].map(sentiment_map)
            dtw_sentiment_numeric = df['DTW_Sentiment'].map(sentiment_map)
            
            ax2.plot(df.index, kmeans_sentiment_numeric, label='KMeans Sentiment', color='blue', alpha=0.5)
            ax2.plot(df.index, dtw_sentiment_numeric, label='DTW Sentiment', color='green', alpha=0.5)
            ax2.set_ylabel('Sentiment')
            ax2.set_yticks([-1, 0, 1])
            ax2.set_yticklabels(['Bearish', 'Sideways', 'Bullish'])
            ax2.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig('strategy_results.png')
    plt.show()

def run_strategy_pipeline(use_csv=True, csv_path='btc_hourly_clustered_extended1.csv', 
                        symbol="BTCUSDT", interval="1h", limit=500, initial_balance=10000):
    """
    Complete end-to-end pipeline to fetch data, run the strategy, analyze performance,
    and visualize results.
    """
    # Step 1: Prepare data 
    df = prepare_data(symbol=symbol, interval=interval, limit=limit, 
                      use_csv=use_csv, csv_path=csv_path)
    
    # Step 2: Execute trading strategy
    df = execute_trade_strategy(df)
    
    # Step 3: Analyze performance
    metrics = analyze_strategy_performance(df, initial_balance)
    
    # Step 4: Print performance metrics
    print_performance_metrics(metrics)
    
    # Step 5: Visualize results
    plot_strategy_results(df, title="RSI-Ichimoku Strategy with Clustering")
    
    # Step 6: Save results
    df.to_csv("strategy_results_with_clusters.csv")
    
    return df, metrics

# Enable direct execution
if __name__ == "__main__":
    # Run the complete pipeline
    df, metrics = run_strategy_pipeline(
        use_csv=True,                         # Set to False to use Binance API directly
        csv_path='btc_hourly_clustered_extended1.csv',  # Path to clustering data
        symbol="BTCUSDT",                     # Trading pair
        interval=Client.KLINE_INTERVAL_1HOUR, # Time interval
        limit=500,                            # Number of records
        initial_balance=10000                 # Initial balance
    )