# Stock Analysis with Machine Learning

This application provides stock analysis using various machine learning algorithms including SVM, Random Forest, KNN, and LSTM. It calculates support and resistance levels and provides trading recommendations based on ML predictions.

## Features

- Real-time stock data fetching using yfinance
- Multiple ML models for prediction:
  - Support Vector Machine (SVM)
  - Random Forest
  - K-Nearest Neighbors (KNN)
  - Long Short-Term Memory (LSTM)
- Support and Resistance level detection
- Trading recommendations with confidence levels
- Interactive web interface

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Start the application:
   ```bash
   python app.py
   ```
2. Open your browser and navigate to `http://localhost:5000`
3. Enter a stock ticker (e.g., AAPL) and click Analyze

## Deployment

To deploy on render.com:

1. Create a new Web Service
2. Connect your repository
3. Set the following:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `gunicorn app:app`

## Technical Details

- Frontend: HTML/TailwindCSS/JavaScript
- Backend: Flask/Python
- ML Libraries: scikit-learn, TensorFlow
- Data Source: Yahoo Finance

## License

MIT