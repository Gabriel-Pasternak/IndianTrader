from flask import Flask, render_template, request, jsonify
from services.stock_service import StockService

app = Flask(__name__)
stock_service = StockService()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/api/analyze', methods=['POST'])
def analyze_stock():
    try:
        data = request.get_json()
        if not data or 'ticker' not in data:
            return jsonify({'error': 'No ticker provided'}), 400
            
        ticker = data.get('ticker')
        result = stock_service.analyze_stock(ticker)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)