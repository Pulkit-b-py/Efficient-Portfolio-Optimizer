Flask app to optimize portfolios of 5 Nifty stocks (RELIANCE, TCS, HDFCBANK, INFY, ICICIBANK) using Kite Connect API. Supports equal-weight, max Sharpe, min risk, and custom strategies with Plotly charts.

## Features
- Pick 2+ stocks
- Strategies: Equal, Max Sharpe, Min Risk, Custom
- Shows returns, risk, Sharpe ratio, pie chart, efficient frontier

## Language
- Python 3.8+
- HTML/CSS/JavaScript (Plotly)

## Files

app.py
templates/index.html
.gitignore
.env.example
requirements.txt
README.md
LICENSE

## Setup
1. **Clone**:
   ```bash
   git clone https://github.com/yourusername/Efficient-Portfolio-Optimizer.git
   cd Efficient-Portfolio-Optimizer

2. **Virtual env**:

python -m venv venv
.\venv\Scripts\activate

3. **Install**:

pip install -r requirements.txt

4. **Set .env**:

copy .env.example .env

5. **Edit .env**:env

KITE_API_KEY=your_api_key

Get key from Kite Developer.

6. **Get token.txt**:

from kiteconnect import KiteConnect

from dotenv import load_dotenv

import os

load_dotenv()

kite = KiteConnect(api_key=os.getenv("KITE_API_KEY"))

print(kite.login_url())  # Open, log in, get request_token

data = kite.generate_session("your_request_token", api_secret="your_api_secret")

with open("token.txt", "w") as f: f.write(data["access_token"])
    
7. **Run**

python app.py

Open http://localhost:5000.

## Usage

Select 2+ stocks

Choose strategy

Click "Analyze" for results and charts

## Security

.env, token.txt ignored by .gitignore

Use own Kite API key/secret

Revoke leaked keys at Kite Developer

