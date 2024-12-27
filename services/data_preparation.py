"""Data preparation utilities for ML models"""
import pandas as pd
import numpy as np
from typing import Tuple

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index"""
    try:
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    except Exception:
        return pd.Series(0, index=prices.index)

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare feature set for ML models"""
    df = df.copy()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['RSI'] = calculate_rsi(df['Close'])
    return df

def prepare_target(df: pd.DataFrame) -> pd.Series:
    """Prepare target variable"""
    return (df['Close'].shift(-1) > df['Close']).astype(int)

def prepare_data(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare both features and target for ML models"""
    features = ['SMA_20', 'SMA_50', 'RSI', 'Volume']
    
    # Prepare features and target
    df_processed = prepare_features(df)
    df_processed['Target'] = prepare_target(df)
    
    # Drop NaN values
    df_processed = df_processed.dropna()
    
    if len(df_processed) < 60:
        raise ValueError("Insufficient data for analysis")
    
    X = df_processed[features].values
    y = df_processed['Target'].values
    
    # Ensure X and y have the same length
    min_len = min(len(X), len(y))
    X = X[:min_len]
    y = y[:min_len]
    
    return X, y
