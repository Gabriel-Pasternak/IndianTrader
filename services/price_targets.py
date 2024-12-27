"""Price target calculation service"""
import pandas as pd
import numpy as np
from typing import List, Dict, Any
from sklearn.linear_model import LinearRegression
from .utils.validation import validate_data_length

class PriceTargetService:
    def __init__(self, ml_service):
        self.ml_service = ml_service

    def get_all_targets(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Get combined price targets from all methods"""
        try:
            validate_data_length(len(df))
            
            targets = []
            
            # Get ML-based targets
            ml_targets = self.ml_service.get_ml_price_targets(df)
            targets.extend(ml_targets)
            
            # Get technical analysis targets
            ta_targets = self._get_technical_targets(df)
            targets.extend(ta_targets)
            
            # Sort and filter unique targets
            return self._filter_unique_targets(targets)[:3]
            
        except Exception as e:
            raise Exception(f"Error calculating price targets: {str(e)}")

    def _get_technical_targets(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Calculate targets using technical analysis"""
        current_price = df['Close'].iloc[-1]
        targets = []
        
        # Add trend-based targets
        targets.extend(self._get_trend_targets(df, current_price))
        
        # Add Fibonacci targets
        targets.extend(self._get_fibonacci_targets(df, current_price))
        
        return targets

    def _get_trend_targets(self, df: pd.DataFrame, current_price: float) -> List[Dict[str, Any]]:
        """Calculate targets using trend analysis"""
        try:
            X = np.arange(len(df)).reshape(-1, 1)
            y = df['Close'].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            future_X = np.array([[len(df)], [len(df) + 5], [len(df) + 20]])
            predictions = model.predict(future_X)
            
            return [
                {
                    'price': float(price),
                    'method': 'Trend',
                    'confidence': self._calculate_confidence(price, current_price)
                }
                for price in predictions
            ]
        except Exception:
            return []

    def _get_fibonacci_targets(self, df: pd.DataFrame, current_price: float) -> List[Dict[str, Any]]:
        """Calculate targets using Fibonacci extensions"""
        try:
            high = df['High'].max()
            low = df['Low'].min()
            range_price = high - low
            
            fib_levels = [1.618, 2.618, 3.618]
            return [
                {
                    'price': float(low + (level * range_price)),
                    'method': f'Fib {level}',
                    'confidence': self._calculate_confidence(low + (level * range_price), current_price)
                }
                for level in fib_levels
            ]
        except Exception:
            return []

    def _calculate_confidence(self, target_price: float, current_price: float) -> float:
        """Calculate confidence score for price target"""
        try:
            price_diff_percent = abs((target_price - current_price) / current_price)
            confidence = max(0, min(100, 100 - (price_diff_percent * 100)))
            return round(float(confidence), 2)
        except Exception:
            return 0.0

    def _filter_unique_targets(self, targets: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter and sort unique targets"""
        try:
            sorted_targets = sorted(targets, key=lambda x: x['price'])
            unique_targets = []
            for target in sorted_targets:
                if not unique_targets or abs(target['price'] - unique_targets[-1]['price']) / unique_targets[-1]['price'] > 0.005:
                    unique_targets.append(target)
            return unique_targets
        except Exception:
            return []
