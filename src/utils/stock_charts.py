"""
Stock chart utilities for the InvestAI Analyst Platform.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import VolumeWeightedAveragePrice
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple


def fetch_stock_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch historical stock data from Yahoo Finance.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format

    Returns:
        DataFrame with historical stock data
    """
    try:
        print(f"Fetching data for {ticker} from {start_date} to {end_date}")

        # Add one day to end_date to include the end date in the results
        end_date_obj = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
        end_date_str = end_date_obj.strftime("%Y-%m-%d")

        # Fetch data
        data = yf.download(ticker, start=start_date, end=end_date_str)

        if data.empty:
            print(f"No data returned for {ticker}")
            return pd.DataFrame()

        print(f"Downloaded data shape: {data.shape}")
        print(f"Data columns: {data.columns.tolist()}")

        # Reset index to make Date a column
        data = data.reset_index()

        # Ensure all price columns are numeric and 1-dimensional
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in data.columns:
                # Convert to numeric if not already
                if not pd.api.types.is_numeric_dtype(data[col]):
                    print(f"Converting {col} to numeric")
                    data[col] = pd.to_numeric(data[col], errors='coerce')

                # Ensure it's 1-dimensional
                if isinstance(data[col], pd.DataFrame) or (isinstance(data[col], np.ndarray) and data[col].ndim > 1):
                    print(f"Flattening {col} from shape {data[col].shape}")
                    data[col] = data[col].iloc[:, 0] if isinstance(data[col], pd.DataFrame) else data[col].flatten()

        return data
    except Exception as e:
        print(f"Error fetching data for {ticker}: {str(e)}")
        import traceback
        traceback.print_exc()
        return pd.DataFrame()


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add technical indicators to the stock data.

    Args:
        df: DataFrame with stock data

    Returns:
        DataFrame with added technical indicators
    """
    if df.empty:
        return df

    # Make a copy to avoid modifying the original
    df = df.copy()

    # Ensure the data is in the correct format
    # Check if Close is a Series or DataFrame column
    if isinstance(df['Close'], pd.DataFrame) or (isinstance(df['Close'], np.ndarray) and df['Close'].ndim > 1):
        print(f"Converting Close column from shape {df['Close'].shape}")
        df['Close'] = df['Close'].iloc[:, 0] if isinstance(df['Close'], pd.DataFrame) else df['Close'].flatten()

    if isinstance(df['High'], pd.DataFrame) or (isinstance(df['High'], np.ndarray) and df['High'].ndim > 1):
        df['High'] = df['High'].iloc[:, 0] if isinstance(df['High'], pd.DataFrame) else df['High'].flatten()

    if isinstance(df['Low'], pd.DataFrame) or (isinstance(df['Low'], np.ndarray) and df['Low'].ndim > 1):
        df['Low'] = df['Low'].iloc[:, 0] if isinstance(df['Low'], pd.DataFrame) else df['Low'].flatten()

    if 'Volume' in df.columns and (isinstance(df['Volume'], pd.DataFrame) or
                                  (isinstance(df['Volume'], np.ndarray) and df['Volume'].ndim > 1)):
        df['Volume'] = df['Volume'].iloc[:, 0] if isinstance(df['Volume'], pd.DataFrame) else df['Volume'].flatten()

    try:
        # Moving Averages
        df['SMA20'] = SMAIndicator(close=df['Close'], window=20).sma_indicator()
        df['SMA50'] = SMAIndicator(close=df['Close'], window=50).sma_indicator()
        df['SMA200'] = SMAIndicator(close=df['Close'], window=200).sma_indicator()
        df['EMA12'] = EMAIndicator(close=df['Close'], window=12).ema_indicator()
        df['EMA26'] = EMAIndicator(close=df['Close'], window=26).ema_indicator()

        # MACD
        macd = MACD(close=df['Close'])
        df['MACD'] = macd.macd()
        df['MACD_Signal'] = macd.macd_signal()
        df['MACD_Hist'] = macd.macd_diff()

        # RSI
        df['RSI'] = RSIIndicator(close=df['Close']).rsi()

        # Stochastic Oscillator
        stoch = StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close'])
        df['Stoch_K'] = stoch.stoch()
        df['Stoch_D'] = stoch.stoch_signal()

        # Bollinger Bands
        bollinger = BollingerBands(close=df['Close'])
        df['BB_High'] = bollinger.bollinger_hband()
        df['BB_Mid'] = bollinger.bollinger_mavg()
        df['BB_Low'] = bollinger.bollinger_lband()

        # VWAP (if we have volume data)
        if 'Volume' in df.columns:
            # VWAP needs a reset index with Date column
            if 'Date' in df.columns:
                vwap = VolumeWeightedAveragePrice(
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    volume=df['Volume'],
                    window=14
                )
                df['VWAP'] = vwap.volume_weighted_average_price()
    except Exception as e:
        print(f"Error calculating indicators: {str(e)}")
        # If there's an error, return the original dataframe without indicators
        return df
    
    return df


def create_stock_chart(
    df: pd.DataFrame,
    ticker: str,
    indicators: List[str] = None,
    height: int = 800
) -> go.Figure:
    """
    Create an interactive stock chart with selected technical indicators.

    Args:
        df: DataFrame with stock data and indicators
        ticker: Stock ticker symbol
        indicators: List of indicators to display
        height: Height of the chart in pixels

    Returns:
        Plotly figure object
    """
    try:
        if df.empty:
            # Return an empty figure with a message
            fig = go.Figure()
            fig.add_annotation(
                text=f"No data available for {ticker}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
            return fig

        # Check if we have the required columns
        required_columns = ['Date', 'Open', 'High', 'Low', 'Close']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            fig = go.Figure()
            fig.add_annotation(
                text=f"Missing required columns: {', '.join(missing_columns)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16, color="red")
            )
            return fig

        # Default indicators if none specified
        if indicators is None:
            indicators = ['SMA20', 'SMA50', 'SMA200', 'BB', 'MACD', 'RSI', 'Volume']

        # Determine how many rows we need for subplots
        n_rows = 3  # Price, MACD, RSI by default
        if 'Volume' in indicators and 'Volume' in df.columns:
            n_rows += 1

        # Create subplot layout
        row_heights = [0.5, 0.15, 0.15]
        if 'Volume' in indicators and 'Volume' in df.columns:
            row_heights.append(0.2)

        fig = make_subplots(
            rows=n_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=row_heights,
            subplot_titles=(
                f"{ticker} Price",
                "MACD",
                "RSI",
                "Volume" if ('Volume' in indicators and 'Volume' in df.columns) else None
            )
        )

        # Add candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=df['Date'],
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name="Price",
                showlegend=False
            ),
            row=1, col=1
        )
    except Exception as e:
        print(f"Error creating chart base: {str(e)}")
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        return fig

        # Add Moving Averages
        if 'SMA20' in indicators and 'SMA20' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['Date'],
                    y=df['SMA20'],
                    name="SMA20",
                    line=dict(color='blue', width=1)
                ),
                row=1, col=1
            )
    
    if 'SMA50' in indicators and 'SMA50' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'], 
                y=df['SMA50'],
                name="SMA50",
                line=dict(color='orange', width=1)
            ),
            row=1, col=1
        )
    
    if 'SMA200' in indicators and 'SMA200' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'], 
                y=df['SMA200'],
                name="SMA200",
                line=dict(color='purple', width=1.5)
            ),
            row=1, col=1
        )
    
    # Add Bollinger Bands
    if 'BB' in indicators and 'BB_High' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'], 
                y=df['BB_High'],
                name="BB Upper",
                line=dict(color='rgba(250, 0, 0, 0.5)', width=1, dash='dash')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['Date'], 
                y=df['BB_Mid'],
                name="BB Middle",
                line=dict(color='rgba(0, 0, 250, 0.5)', width=1, dash='dash')
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['Date'], 
                y=df['BB_Low'],
                name="BB Lower",
                line=dict(color='rgba(0, 250, 0, 0.5)', width=1, dash='dash'),
                fill='tonexty', 
                fillcolor='rgba(200, 200, 200, 0.1)'
            ),
            row=1, col=1
        )
    
    # Add MACD
    if 'MACD' in indicators and 'MACD' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'], 
                y=df['MACD'],
                name="MACD",
                line=dict(color='blue', width=1.5)
            ),
            row=2, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=df['Date'], 
                y=df['MACD_Signal'],
                name="Signal",
                line=dict(color='red', width=1)
            ),
            row=2, col=1
        )
        
        # Add MACD histogram
        colors = ['green' if val >= 0 else 'red' for val in df['MACD_Hist']]
        fig.add_trace(
            go.Bar(
                x=df['Date'], 
                y=df['MACD_Hist'],
                name="Histogram",
                marker_color=colors,
                showlegend=False
            ),
            row=2, col=1
        )
        
        # Add a zero line
        fig.add_shape(
            type="line",
            x0=df['Date'].iloc[0],
            y0=0,
            x1=df['Date'].iloc[-1],
            y1=0,
            line=dict(color="gray", width=1, dash="dash"),
            row=2, col=1
        )
    
    # Add RSI
    if 'RSI' in indicators and 'RSI' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df['Date'], 
                y=df['RSI'],
                name="RSI",
                line=dict(color='purple', width=1.5)
            ),
            row=3, col=1
        )
        
        # Add overbought/oversold lines
        fig.add_shape(
            type="line",
            x0=df['Date'].iloc[0],
            y0=70,
            x1=df['Date'].iloc[-1],
            y1=70,
            line=dict(color="red", width=1, dash="dash"),
            row=3, col=1
        )
        
        fig.add_shape(
            type="line",
            x0=df['Date'].iloc[0],
            y0=30,
            x1=df['Date'].iloc[-1],
            y1=30,
            line=dict(color="green", width=1, dash="dash"),
            row=3, col=1
        )
        
        # Add a middle line
        fig.add_shape(
            type="line",
            x0=df['Date'].iloc[0],
            y0=50,
            x1=df['Date'].iloc[-1],
            y1=50,
            line=dict(color="gray", width=1, dash="dash"),
            row=3, col=1
        )
    
    # Add Volume
    if 'Volume' in indicators and 'Volume' in df.columns:
        # Color volume bars based on price change
        colors = ['green' if df['Close'].iloc[i] >= df['Open'].iloc[i] else 'red' 
                 for i in range(len(df))]
        
        fig.add_trace(
            go.Bar(
                x=df['Date'], 
                y=df['Volume'],
                name="Volume",
                marker_color=colors,
                showlegend=False
            ),
            row=4 if n_rows == 4 else 3, col=1
        )
    
    # Update layout
    fig.update_layout(
        title=f"{ticker} Stock Analysis",
        xaxis_title="Date",
        yaxis_title="Price ($)",
        height=height,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_rangeslider_visible=False,
        margin=dict(l=50, r=50, t=80, b=50),
    )
    
    # Update y-axis labels
    fig.update_yaxes(title_text="Price ($)", row=1, col=1)
    fig.update_yaxes(title_text="MACD", row=2, col=1)
    fig.update_yaxes(title_text="RSI", row=3, col=1)
    if 'Volume' in indicators and n_rows == 4:
        fig.update_yaxes(title_text="Volume", row=4, col=1)
    
    # Set RSI range
    fig.update_yaxes(range=[0, 100], row=3, col=1)
    
    return fig


def get_stock_chart_with_indicators(
    ticker: str,
    start_date: str,
    end_date: str,
    indicators: List[str] = None
) -> go.Figure:
    """
    Fetch stock data and create an interactive chart with technical indicators.

    Args:
        ticker: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        indicators: List of indicators to display

    Returns:
        Plotly figure object
    """
    try:
        # Fetch data
        df = fetch_stock_data(ticker, start_date, end_date)

        if df.empty:
            # Return an empty figure with a message
            fig = go.Figure()
            fig.add_annotation(
                text=f"No data available for {ticker}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=20)
            )
            return fig

        # Print data info for debugging
        print(f"Data fetched for {ticker}: {len(df)} rows")
        print(f"Columns: {df.columns.tolist()}")
        for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
            if col in df.columns:
                print(f"{col} column type: {type(df[col])}, shape: {df[col].shape if hasattr(df[col], 'shape') else 'N/A'}")

        # Add technical indicators
        df = add_technical_indicators(df)

        # Create chart
        fig = create_stock_chart(df, ticker, indicators)

        return fig
    except Exception as e:
        # If any error occurs, return a figure with the error message
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error creating chart for {ticker}: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=16, color="red")
        )
        # Re-raise the exception for debugging
        raise