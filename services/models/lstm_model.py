from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import numpy as np
from typing import Tuple

def create_lstm_model() -> Sequential:
    """Create and compile LSTM model"""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    return model

def prepare_lstm_data(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare data specifically for LSTM model"""
    if len(X) < 60:
        raise ValueError("Insufficient data for LSTM")
    X_lstm = X[-60:].reshape((1, 60, 1))
    y_lstm = np.array([y[-1]])
    return X_lstm, y_lstm