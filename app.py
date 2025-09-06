"""
EFFICIENT PORTFOLIO OPTIMIZER - Web Application
Analyzes 5 Blue-chip Nifty Stocks for Optimal Portfolio Allocation
"""

from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy.optimize import minimize
import json
import plotly.graph_objs as go
import plotly.utils
from kiteconnect import KiteConnect
import os

app = Flask(__name__)

# ========== CONFIGURATION ==========
# Top 5 Blue-chip Nifty Stocks with their NSE tokens
STOCKS = {
    'RELIANCE': {'token': 738561, 'name': 'Reliance Industries'},
    'TCS': {'token': 2953217, 'name': 'Tata Consultancy Services'},
    'HDFCBANK': {'token': 341249, 'name': 'HDFC Bank'},
    'INFY': {'token': 408065, 'name': 'Infosys'},
    'ICICIBANK': {'token': 1270529, 'name': 'ICICI Bank'}
}

# Load Kite connection
def get_kite_connection():
    """Load existing Kite connection from saved token"""
    try:
        if not os.path.exists("token.txt"):
            raise FileNotFoundError("token.txt not found. Please generate an access token.")
        with open("token.txt", "r") as f:
            access_token = f.read().strip()
        API_KEY = os.getenv("KITE_API_KEY", "5igxd0bd1z16zc2p")
        kite = KiteConnect(api_key=API_KEY)
        kite.set_access_token(access_token)
        kite.instruments()  # Verify connection
        return kite
    except Exception as e:
        print(f"Error loading Kite connection: {e}")
        return None

# Global Kite instance
kite = get_kite_connection()

# ========== DATA FETCHING ==========
def fetch_stock_data(stock_token, from_date, to_date):
    """Fetch historical data for a stock"""
    try:
        all_data = []
        current_date = from_date
        while current_date < to_date:
            chunk_end = min(current_date + timedelta(days=1999), to_date)
            chunk_data = kite.historical_data(
                stock_token, 
                current_date, 
                chunk_end, 
                "day"
            )
            if not chunk_data:
                print(f"No data returned for token {stock_token}")
                return None
            all_data.extend(chunk_data)
            current_date = chunk_end + timedelta(days=1)
        df = pd.DataFrame(all_data)
        if df.empty:
            print(f"Empty data for token {stock_token}")
            return None
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        return df
    except Exception as e:
        print(f"Error fetching data for token {stock_token}: {e}")
        return None

def prepare_portfolio_data():
    """Fetch and prepare data for all stocks"""
    from_date = datetime(2023, 1, 1)
    to_date = datetime.now()
    
    stock_data = {}
    returns_data = {}
    
    for symbol, info in STOCKS.items():
        print(f"Fetching {symbol}...")
        df = fetch_stock_data(info['token'], from_date, to_date)
        if df is not None and not df.empty:
            stock_data[symbol] = df
            returns_data[symbol] = df['close'].pct_change().dropna()
    
    # Create combined returns DataFrame
    returns_df = pd.DataFrame(returns_data).dropna()
    
    return stock_data, returns_df

# ========== PORTFOLIO OPTIMIZATION ==========
def calculate_portfolio_metrics(weights, returns_df):
    """Calculate portfolio return, risk, and Sharpe ratio"""
    portfolio_return = np.sum(returns_df.mean() * weights) * 252
    portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns_df.cov() * 252, weights)))
    sharpe_ratio = (portfolio_return - 0.06) / portfolio_std  # 6% risk-free rate
    
    return {
        'return': float(portfolio_return * 100),
        'risk': float(portfolio_std * 100),
        'sharpe': float(sharpe_ratio)
    }

def optimize_portfolio(returns_df, optimization_type='sharpe'):
    """Find optimal portfolio weights"""
    n_assets = len(returns_df.columns)
    
    # Constraints and bounds
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))
    initial_weights = np.array([1/n_assets] * n_assets)
    
    if optimization_type == 'sharpe':
        # Maximize Sharpe ratio
        def negative_sharpe(weights):
            metrics = calculate_portfolio_metrics(weights, returns_df)
            return -metrics['sharpe']
        
        result = minimize(negative_sharpe, initial_weights, 
                         method='SLSQP', bounds=bounds, constraints=constraints)
    
    elif optimization_type == 'min_risk':
        # Minimize risk
        def portfolio_risk(weights):
            metrics = calculate_portfolio_metrics(weights, returns_df)
            return metrics['risk']
        
        result = minimize(portfolio_risk, initial_weights,
                         method='SLSQP', bounds=bounds, constraints=constraints)
    
    else:  # Custom weights
        return None
    
    if result.success:
        return result.x
    return None

def generate_efficient_frontier(returns_df, n_portfolios=50):
    """Generate efficient frontier data"""
    n_assets = len(returns_df.columns)
    
    # Generate random portfolios
    np.random.seed(42)
    random_weights = []
    random_returns = []
    random_risks = []
    random_sharpes = []
    
    for _ in range(1000):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)
        metrics = calculate_portfolio_metrics(weights, returns_df)
        
        random_weights.append(weights)
        random_returns.append(metrics['return'])
        random_risks.append(metrics['risk'])
        random_sharpes.append(metrics['sharpe'])
    
    # Find optimal portfolios
    max_sharpe_weights = optimize_portfolio(returns_df, 'sharpe')
    min_risk_weights = optimize_portfolio(returns_df, 'min_risk')
    
    max_sharpe_metrics = calculate_portfolio_metrics(max_sharpe_weights, returns_df)
    min_risk_metrics = calculate_portfolio_metrics(min_risk_weights, returns_df)
    
    # Generate frontier curve
    target_returns = np.linspace(min(random_returns), max(random_returns), n_portfolios)
    frontier_risks = []
    frontier_weights = []
    
    for target_return in target_returns:
        constraints = [
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: calculate_portfolio_metrics(x, returns_df)['return'] - target_return}
        ]
        
        result = minimize(
            lambda w: calculate_portfolio_metrics(w, returns_df)['risk'],
            np.array([1/n_assets] * n_assets),
            method='SLSQP',
            bounds=tuple((0, 1) for _ in range(n_assets)),
            constraints=constraints
        )
        
        if result.success:
            frontier_risks.append(calculate_portfolio_metrics(result.x, returns_df)['risk'])
            frontier_weights.append(result.x)
        else:
            frontier_risks.append(None)
            frontier_weights.append(None)
    
    return {
        'random': {
            'returns': [float(r) for r in random_returns],
            'risks': [float(r) for r in random_risks],
            'sharpes': [float(s) for s in random_sharpes]
        },
        'frontier': {
            'returns': [float(r) for r in target_returns],
            'risks': [float(r) if r is not None else None for r in frontier_risks]
        },
        'optimal': {
            'max_sharpe': {
                'weights': [float(w) for w in max_sharpe_weights.tolist()],
                'metrics': {
                    'return': float(max_sharpe_metrics['return']),
                    'risk': float(max_sharpe_metrics['risk']),
                    'sharpe': float(max_sharpe_metrics['sharpe'])
                }
            },
            'min_risk': {
                'weights': [float(w) for w in min_risk_weights.tolist()],
                'metrics': {
                    'return': float(min_risk_metrics['return']),
                    'risk': float(min_risk_metrics['risk']),
                    'sharpe': float(min_risk_metrics['sharpe'])
                }
            }
        }
    }

# ========== WEB ROUTES ==========
@app.route('/')
def index():
    """Main page"""
    return render_template('index.html', stocks=STOCKS)

@app.route('/analyze', methods=['POST'])
def analyze_portfolio():
    """Analyze portfolio based on user inputs"""
    try:
        data = request.json
        selected_stocks = data.get('stocks', list(STOCKS.keys()))
        weights_input = data.get('weights', {})
        optimization = data.get('optimization', 'equal')

        # Validate selected stocks
        if len(selected_stocks) < 2:
            return jsonify({'success': False, 'error': 'At least 2 stocks must be selected'})

        # Fetch and prepare data
        print("Fetching stock data...")
        stock_data, returns_df = prepare_portfolio_data()
        if returns_df.empty:
            return jsonify({'success': False, 'error': 'No valid stock data available'})

        # Filter selected stocks
        returns_df = returns_df[selected_stocks]

        # Determine weights based on optimization type
        if optimization == 'max_sharpe':
            weights = optimize_portfolio(returns_df, 'sharpe')
        elif optimization == 'min_risk':
            weights = optimize_portfolio(returns_df, 'min_risk')
        elif optimization == 'custom':
            weights = np.array([weights_input.get(stock, 0) for stock in selected_stocks])
            if np.sum(weights) == 0:
                return jsonify({'success': False, 'error': 'Custom weights must sum to a positive value'})
            weights = weights / np.sum(weights)  # Normalize
        else:  # equal
            weights = np.array([1/len(selected_stocks)] * len(selected_stocks))

        # Check if optimization succeeded
        if weights is None:
            return jsonify({'success': False, 'error': 'Optimization failed or invalid optimization type'})

        # Calculate portfolio metrics
        metrics = calculate_portfolio_metrics(weights, returns_df)
        metrics = {
            'return': float(metrics['return']),
            'risk': float(metrics['risk']),
            'sharpe': float(metrics['sharpe'])
        }

        # Generate efficient frontier
        frontier_data = generate_efficient_frontier(returns_df)

        # Create portfolio allocation chart
        allocation_chart = {
            'data': [{
                'type': 'pie',
                'labels': selected_stocks,
                'values': [float(w * 100) for w in weights],  # Convert to float
                'hole': 0.4,
                'textinfo': 'label+percent',
                'textposition': 'outside'
            }],
            'layout': {
                'title': 'Portfolio Allocation',
                'height': 400
            }
        }

        # Create efficient frontier chart
        frontier_chart = {
            'data': [
                {
                    'type': 'scatter',
                    'x': [float(x) for x in frontier_data['random']['risks']],
                    'y': [float(y) for y in frontier_data['random']['returns']],
                    'mode': 'markers',
                    'name': 'Random Portfolios',
                    'marker': {
                        'size': 5,
                        'color': [float(c) for c in frontier_data['random']['sharpes']],
                        'colorscale': 'Viridis',
                        'showscale': True,
                        'colorbar': {'title': 'Sharpe Ratio'}
                    }
                },
                {
                    'type': 'scatter',
                    'x': [float(x) if x is not None else None for x in frontier_data['frontier']['risks']],
                    'y': [float(y) for y in frontier_data['frontier']['returns']],
                    'mode': 'lines',
                    'name': 'Efficient Frontier',
                    'line': {'color': 'red', 'width': 3}
                },
                {
                    'type': 'scatter',
                    'x': [float(frontier_data['optimal']['max_sharpe']['metrics']['risk'])],
                    'y': [float(frontier_data['optimal']['max_sharpe']['metrics']['return'])],
                    'mode': 'markers',
                    'name': 'Max Sharpe',
                    'marker': {'size': 15, 'color': 'red', 'symbol': 'star'}
                },
                {
                    'type': 'scatter',
                    'x': [float(frontier_data['optimal']['min_risk']['metrics']['risk'])],
                    'y': [float(frontier_data['optimal']['min_risk']['metrics']['return'])],
                    'mode': 'markers',
                    'name': 'Min Risk',
                    'marker': {'size': 15, 'color': 'green', 'symbol': 'star'}
                },
                {
                    'type': 'scatter',
                    'x': [float(metrics['risk'])],
                    'y': [float(metrics['return'])],
                    'mode': 'markers',
                    'name': 'Your Portfolio',
                    'marker': {'size': 15, 'color': 'blue', 'symbol': 'diamond'}
                }
            ],
            'layout': {
                'title': 'Efficient Frontier Analysis',
                'xaxis': {'title': 'Risk (Annual Volatility %)'},
                'yaxis': {'title': 'Return (Annual %)'},
                'height': 500,
                'hovermode': 'closest'
            }
        }

        # Prepare response
        response = {
            'success': True,
            'portfolio': {
                'weights': {stock: float(weight) for stock, weight in zip(selected_stocks, weights.tolist())},
                'metrics': metrics
            },
            'optimal': frontier_data['optimal'],
            'charts': {
                'allocation': allocation_chart,
                'frontier': frontier_chart
            }
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_stock_performance', methods=['POST'])
def get_stock_performance():
    """Get individual stock performance"""
    try:
        # Fetch data
        stock_data, returns_df = prepare_portfolio_data()
        if returns_df.empty:
            return jsonify({'success': False, 'error': 'No valid stock data available'})
        
        performance = {}
        for symbol in STOCKS.keys():
            if symbol in returns_df.columns:
                returns = returns_df[symbol]
                annual_return = returns.mean() * 252 * 100
                annual_vol = returns.std() * np.sqrt(252) * 100
                sharpe = (annual_return - 6) / annual_vol
                
                performance[symbol] = {
                    'return': float(round(annual_return, 2)),
                    'risk': float(round(annual_vol, 2)),
                    'sharpe': float(round(sharpe, 3))
                }
        
        return jsonify({'success': True, 'performance': performance})
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

# ========== HTML TEMPLATE ==========
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Efficient Portfolio Optimizer</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container {
            max-width: 1400px;
            margin: 0 auto;
            background: white;
            border-radius: 20px;
            box-shadow: 0 20px 60px rgba(0,0,0,0.3);
            overflow: hidden;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            text-align: center;
        }
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
        }
        .header p {
            font-size: 1.1em;
            opacity: 0.9;
        }
        .main-content {
            display: grid;
            grid-template-columns: 350px 1fr;
            gap: 30px;
            padding: 30px;
        }
        .sidebar {
            background: #f8f9fa;
            padding: 25px;
            border-radius: 15px;
            height: fit-content;
        }
        .sidebar h2 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.4em;
        }
        .stock-selector {
            margin-bottom: 25px;
        }
        .stock-item {
            background: white;
            padding: 12px;
            margin-bottom: 10px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            transition: all 0.3s;
            cursor: pointer;
        }
        .stock-item:hover {
            transform: translateX(5px);
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        }
        .stock-item input[type="checkbox"] {
            margin-right: 10px;
            width: 18px;
            height: 18px;
        }
        .stock-item label {
            flex: 1;
            cursor: pointer;
            font-weight: 500;
        }
        .optimization-section {
            margin-bottom: 25px;
        }
        .optimization-section h3 {
            color: #555;
            margin-bottom: 15px;
            font-size: 1.1em;
        }
        .radio-group {
            display: flex;
            flex-direction: column;
            gap: 10px;
        }
        .radio-item {
            background: white;
            padding: 12px;
            border-radius: 8px;
            display: flex;
            align-items: center;
            cursor: pointer;
            transition: all 0.3s;
        }
        .radio-item:hover {
            background: #e9ecef;
        }
        .radio-item input[type="radio"] {
            margin-right: 10px;
        }
        .radio-item label {
            cursor: pointer;
            flex: 1;
        }
        .custom-weights {
            margin-top: 15px;
            padding: 15px;
            background: white;
            border-radius: 8px;
            display: none;
        }
        .weight-input {
            display: flex;
            align-items: center;
            margin-bottom: 10px;
        }
        .weight-input label {
            flex: 1;
            font-size: 0.9em;
        }
        .weight-input input {
            width: 80px;
            padding: 5px;
            border: 1px solid #ddd;
            border-radius: 4px;
            text-align: right;
        }
        .analyze-btn {
            width: 100%;
            padding: 15px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.1em;
            font-weight: bold;
            cursor: pointer;
            transition: all 0.3s;
        }
        .analyze-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 20px rgba(0,0,0,0.2);
        }
        .results-section {
            background: white;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin-bottom: 30px;
        }
        .metric-card {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            padding: 20px;
            border-radius: 12px;
            text-align: center;
        }
        .metric-card h3 {
            color: #555;
            font-size: 0.9em;
            margin-bottom: 10px;
            text-transform: uppercase;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #333;
        }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        }
        .loading {
            text-align: center;
            padding: 50px;
            font-size: 1.2em;
            color: #666;
        }
        .recommendations {
            background: #f8f9fa;
            padding: 20px;
            border-radius: 12px;
            margin-top: 20px;
        }
        .recommendations h3 {
            color: #333;
            margin-bottom: 15px;
        }
        .recommendation-item {
            background: white;
            padding: 15px;
            border-radius: 8px;
            margin-bottom: 10px;
            border-left: 4px solid #667eea;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .spinner {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #667eea;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
            margin: 20px auto;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>📈 Efficient Portfolio Optimizer</h1>
            <p>Analyze Blue-chip Nifty Stocks for Optimal Portfolio Allocation</p>
        </div>
        
        <div class="main-content">
            <div class="sidebar">
                <h2>Portfolio Configuration</h2>
                
                <div class="stock-selector">
                    <h3>Select Stocks</h3>
                    <div id="stockList">
                        <!-- Stocks will be populated here -->
                    </div>
                </div>
                
                <div class="optimization-section">
                    <h3>Optimization Strategy</h3>
                    <div class="radio-group">
                        <div class="radio-item">
                            <input type="radio" id="equal" name="optimization" value="equal" checked>
                            <label for="equal">Equal Weights</label>
                        </div>
                        <div class="radio-item">
                            <input type="radio" id="max_sharpe" name="optimization" value="max_sharpe">
                            <label for="max_sharpe">Maximum Sharpe Ratio</label>
                        </div>
                        <div class="radio-item">
                            <input type="radio" id="min_risk" name="optimization" value="min_risk">
                            <label for="min_risk">Minimum Risk</label>
                        </div>
                        <div class="radio-item">
                            <input type="radio" id="custom" name="optimization" value="custom">
                            <label for="custom">Custom Weights</label>
                        </div>
                    </div>
                    
                    <div class="custom-weights" id="customWeights">
                        <!-- Custom weight inputs will be populated here -->
                    </div>
                </div>
                
                <button class="analyze-btn" onclick="analyzePortfolio()">
                    Analyze Portfolio
                </button>
            </div>
            
            <div class="results-section">
                <div id="results">
                    <div class="loading">
                        Select stocks and click "Analyze Portfolio" to begin
                    </div>
                </div>
            </div>
        </div>
    </div>
    
    <script>
        // Stock data
        const stocks = {{ stocks | tojson }};
        
        // Initialize stock list
        function initializeStocks() {
            const stockList = document.getElementById('stockList');
            const customWeights = document.getElementById('customWeights');
            
            for (const [symbol, info] of Object.entries(stocks)) {
                // Add to stock selector
                const stockItem = document.createElement('div');
                stockItem.className = 'stock-item';
                stockItem.innerHTML = `
                    <input type="checkbox" id="${symbol}" value="${symbol}" checked>
                    <label for="${symbol}">${symbol} - ${info.name}</label>
                `;
                stockList.appendChild(stockItem);
                
                // Add to custom weights
                const weightInput = document.createElement('div');
                weightInput.className = 'weight-input';
                weightInput.innerHTML = `
                    <label for="weight_${symbol}">${symbol}</label>
                    <input type="number" id="weight_${symbol}" value="20" min="0" max="100" step="1">%
                `;
                customWeights.appendChild(weightInput);
            }
        }
        
        // Handle optimization strategy change
        document.querySelectorAll('input[name="optimization"]').forEach(radio => {
            radio.addEventListener('change', function() {
                const customWeights = document.getElementById('customWeights');
                customWeights.style.display = this.value === 'custom' ? 'block' : 'none';
            });
        });
        
        // Analyze portfolio
        async function analyzePortfolio() {
            // Get selected stocks
            const selectedStocks = [];
            document.querySelectorAll('#stockList input[type="checkbox"]:checked').forEach(checkbox => {
                selectedStocks.push(checkbox.value);
            });
            
            if (selectedStocks.length < 2) {
                alert('Please select at least 2 stocks for portfolio analysis');
                return;
            }
            
            // Get optimization type
            const optimization = document.querySelector('input[name="optimization"]:checked').value;
            
            // Get custom weights if applicable
            const weights = {};
            if (optimization === 'custom') {
                selectedStocks.forEach(stock => {
                    weights[stock] = parseFloat(document.getElementById(`weight_${stock}`).value) || 0;
                });
            }
            
            // Show loading
            const resultsDiv = document.getElementById('results');
            resultsDiv.innerHTML = '<div class="loading"><div class="spinner"></div>Analyzing portfolio...</div>';
            
            try {
                // Send request to analyze
                const response = await fetch('/analyze', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({
                        stocks: selectedStocks,
                        weights: weights,
                        optimization: optimization
                    })
                });
                
                const data = await response.json();
                
                if (data.success) {
                    displayResults(data);
                } else {
                    resultsDiv.innerHTML = `<div class="loading">Error: ${data.error}</div>`;
                }
            } catch (error) {
                resultsDiv.innerHTML = `<div class="loading">Error: ${error.message}</div>`;
            }
        }
        
        // Display results
        function displayResults(data) {
            const resultsDiv = document.getElementById('results');
            
            // Create metrics cards
            const metricsHTML = `
                <div class="metrics-grid">
                    <div class="metric-card">
                        <h3>Expected Return</h3>
                        <div class="metric-value">${data.portfolio.metrics.return.toFixed(2)}%</div>
                    </div>
                    <div class="metric-card">
                        <h3>Annual Risk</h3>
                        <div class="metric-value">${data.portfolio.metrics.risk.toFixed(2)}%</div>
                    </div>
                    <div class="metric-card">
                        <h3>Sharpe Ratio</h3>
                        <div class="metric-value">${data.portfolio.metrics.sharpe.toFixed(3)}</div>
                    </div>
                </div>
            `;
            
            // Create allocation table
            let allocationHTML = '<div class="chart-container"><h3>Portfolio Allocation</h3><table style="width:100%">';
            for (const [stock, weight] of Object.entries(data.portfolio.weights)) {
                allocationHTML += `
                    <tr>
                        <td>${stock}</td>
                        <td style="text-align:right">${(weight * 100).toFixed(2)}%</td>
                    </tr>
                `;
            }
            allocationHTML += '</table></div>';
            
            // Create charts
            const chartsHTML = `
                <div class="chart-container">
                    <div id="allocationChart"></div>
                </div>
                <div class="chart-container">
                    <div id="frontierChart"></div>
                </div>
            `;
            
            // Create recommendations
            const recoHTML = `
                <div class="recommendations">
                    <h3>Recommendations</h3>
                    <div class="recommendation-item">
                        <strong>Optimal Portfolio (Max Sharpe):</strong><br>
                        Return: ${data.optimal.max_sharpe.metrics.return.toFixed(2)}%, 
                        Risk: ${data.optimal.max_sharpe.metrics.risk.toFixed(2)}%, 
                        Sharpe: ${data.optimal.max_sharpe.metrics.sharpe.toFixed(3)}
                    </div>
                    <div class="recommendation-item">
                        <strong>Conservative Portfolio (Min Risk):</strong><br>
                        Return: ${data.optimal.min_risk.metrics.return.toFixed(2)}%, 
                        Risk: ${data.optimal.min_risk.metrics.risk.toFixed(2)}%, 
                        Sharpe: ${data.optimal.min_risk.metrics.sharpe.toFixed(3)}
                    </div>
                </div>
            `;
            
            // Update results
            resultsDiv.innerHTML = metricsHTML + chartsHTML + allocationHTML + recoHTML;
            
            // Render charts
            Plotly.newPlot('allocationChart', data.charts.allocation.data, data.charts.allocation.layout);
            Plotly.newPlot('frontierChart', data.charts.frontier.data, data.charts.frontier.layout);
        }
        
        // Initialize on load
        initializeStocks();
        
        // Load initial performance
        fetch('/get_stock_performance', {method: 'POST'})
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    console.log('Stock performance loaded:', data.performance);
                }
            });
    </script>
</body>
</html>
"""

# Create templates directory and save HTML
os.makedirs('templates', exist_ok=True)
with open('templates/index.html', 'w', encoding='utf-8') as f:
    f.write(HTML_TEMPLATE)

# ========== RUN APPLICATION ==========
if __name__ == '__main__':
    print("\n" + "="*60)
    print("🚀 EFFICIENT PORTFOLIO OPTIMIZER")
    print("="*60)
    print("\n📊 Analyzing 5 Blue-chip Stocks:")
    for symbol, info in STOCKS.items():
        print(f"   • {symbol}: {info['name']}")
    
    print("\n⚙️ Starting web server...")
    print("\n✅ Server running at: http://localhost:5000")
    print("\n📌 Instructions:")
    print("   1. Open your browser and go to http://localhost:5000")
    print("   2. Select stocks for your portfolio")
    print("   3. Choose optimization strategy")
    print("   4. Click 'Analyze Portfolio' to see results")
    print("\n⚠️ Press Ctrl+C to stop the server")
    print("="*60 + "\n")
    
    app.run(debug=True, port=5000)

    from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("API_KEY")