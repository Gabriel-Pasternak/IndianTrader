import yfinance as yf
import pandas as pd
from typing import List, Dict, Any
from .price_targets import PriceTargetService
from .ml_service import MLService

class StockService:
    def __init__(self):
        self.ml_service = MLService()
        self.price_target_service = PriceTargetService(self.ml_service)

    def get_stock_data(self, ticker: str, period: str = '1y') -> pd.DataFrame:
        """Fetch stock data from Yahoo Finance"""
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            if df.empty:
                raise ValueError(f"No data found for ticker {ticker}")
            return df
        except Exception as e:
            raise Exception(f"Error fetching stock data: {str(e)}")

    def analyze_stock(self, ticker: str) -> Dict[str, Any]:
        """Complete stock analysis"""
        try:
            # Get stock data
            df = self.get_stock_data(ticker)
            
            # Get ML predictions
            predictions = self.ml_service.get_predictions(df)
            
            # Get support and resistance levels
            levels = self.get_support_resistance_levels(df)
            
            # Get price targets
            targets = self.price_target_service.get_all_targets(df)
            
            return {
                'predictions': predictions,
                'levels': levels,
                'targets': targets,
                'recommendations': self.ml_service.get_recommendations(predictions)
            }
        except Exception as e:
            raise Exception(f"Error analyzing stock: {str(e)}")

    def get_support_resistance_levels(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Calculate support and resistance levels"""
        try:
            if df.empty:
                return []

            levels = []
            for i in range(2, len(df)-2):
                if self._is_support(df, i):
                    levels.append({
                        'type': 'support',
                        'price': float(df['Low'].iloc[i]),
                        'confidence': self._calculate_level_confidence(df, df['Low'].iloc[i])
                    })
                elif self._is_resistance(df, i):
                    levels.append({
                        'type': 'resistance',
                        'price': float(df['High'].iloc[i]),
                        'confidence': self._calculate_level_confidence(df, df['High'].iloc[i])
                    })
            
            levels.sort(key=lambda x: x['confidence'], reverse=True)
            return levels[:5]
        
        except Exception as e:
            raise Exception(f"Error calculating support/resistance levels: {str(e)}")

    def _is_support(self, df: pd.DataFrame, i: int) -> bool:
        """Check if index i is a support level"""
        return (df['Low'].iloc[i] < df['Low'].iloc[i-1] and 
                df['Low'].iloc[i] < df['Low'].iloc[i+1] and 
                df['Low'].iloc[i+1] < df['Low'].iloc[i+2] and 
                df['Low'].iloc[i-1] < df['Low'].iloc[i-2])

    def _is_resistance(self, df: pd.DataFrame, i: int) -> bool:
        """Check if index i is a resistance level"""
        return (df['High'].iloc[i] > df['High'].iloc[i-1] and 
                df['High'].iloc[i] > df['High'].iloc[i+1] and 
                df['High'].iloc[i+1] > df['High'].iloc[i+2] and 
                df['High'].iloc[i-1] > df['High'].iloc[i-2])

    def _calculate_level_confidence(self, df: pd.DataFrame, level_price: float) -> float:
        """Calculate confidence score for support/resistance level"""
        try:
            price_range = df['High'].max() - df['Low'].min()
            if price_range == 0:
                return 0
            touches = sum(abs(df['Close'] - level_price) < price_range * 0.02)
            return min(float(touches * 10), 100)
        except Exception:
            return 0