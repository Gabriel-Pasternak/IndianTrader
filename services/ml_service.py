"""Machine learning service for stock analysis"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, List
from .data_preparation import prepare_data
from .models import create_lstm_model, prepare_lstm_data
from .utils.validation import validate_data_length

class MLService:
    def __init__(self):
        self.models = {
            'rf': RandomForestClassifier(n_estimators=100, random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'knn': KNeighborsClassifier(n_neighbors=5)
        }
        self.scaler = StandardScaler()

    def get_predictions(self, df) -> Dict[str, float]:
        """Get predictions from all models"""
        try:
            validate_data_length(len(df))
            X, y = prepare_data(df)
            X_scaled = self.scaler.fit_transform(X)
            
            predictions = {}
            
            # Traditional ML models
            for name, model in self.models.items():
                try:
                    model.fit(X_scaled[:-1], y[:-1])
                    pred = model.predict_proba(X_scaled[-1:])[:, 1]
                    predictions[name] = float(pred[0])
                except Exception as e:
                    print(f"Error in {name} model: {str(e)}")
                    predictions[name] = 0.5
            
            # LSTM model
            try:
                lstm = create_lstm_model()
                X_lstm, y_lstm = prepare_lstm_data(X_scaled, y)
                lstm.fit(X_lstm, y_lstm, epochs=50, verbose=0)
                pred = lstm.predict(X_lstm[-1:])
                predictions['lstm'] = float(pred[0][0])
            except Exception as e:
                print(f"Error in LSTM model: {str(e)}")
                predictions['lstm'] = 0.5
            
            return predictions
            
        except Exception as e:
            raise Exception(f"Error getting predictions: {str(e)}")

    def get_ml_price_targets(self, df) -> List[Dict[str, Any]]:
        """Generate price targets using ML models"""
        try:
            current_price = df['Close'].iloc[-1]
            avg_daily_move = df['Close'].pct_change().std()
            
            predictions = self.get_predictions(df)
            avg_confidence = np.mean(list(predictions.values()))
            
            # Calculate potential price moves based on ML confidence
            if avg_confidence > 0.5:  # Bullish
                move_multiplier = avg_confidence * 2  # More confident = bigger move
            else:  # Bearish
                move_multiplier = (1 - avg_confidence) * 2
                
            targets = []
            for days in [1, 5, 20]:  # Short, medium, long term
                expected_move = current_price * (avg_daily_move * move_multiplier * np.sqrt(days))
                if avg_confidence > 0.5:
                    target_price = current_price + expected_move
                else:
                    target_price = current_price - expected_move
                    
                targets.append({
                    'price': float(target_price),
                    'method': f'ML {days}D',
                    'confidence': float(abs(avg_confidence - 0.5) * 200)  # Convert to 0-100 scale
                })
                
            return targets
            
        except Exception as e:
            raise Exception(f"Error calculating ML price targets: {str(e)}")

    def get_recommendations(self, predictions: Dict[str, float]) -> Dict[str, Any]:
        """Generate trading recommendations based on model predictions"""
        try:
            avg_confidence = np.mean(list(predictions.values()))
            
            if avg_confidence > 0.7:
                action = "Strong Buy"
                reason = "High confidence signals from multiple models"
            elif avg_confidence > 0.6:
                action = "Buy"
                reason = "Moderate positive signals"
            elif avg_confidence < 0.3:
                action = "Strong Sell"
                reason = "Strong negative signals from multiple models"
            elif avg_confidence < 0.4:
                action = "Sell"
                reason = "Moderate negative signals"
            else:
                action = "Hold"
                reason = "Mixed signals from models"
                
            return {
                'action': action,
                'reason': reason,
                'confidence': round(avg_confidence * 100, 2)
            }
        except Exception as e:
            return {
                'action': 'Error',
                'reason': str(e),
                'confidence': 0
            }
