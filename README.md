# Efficient-Portfolio-Optimizer
This project is a Flask web application for building and analyzing efficient investment portfolios. It integrates with market data, uses optimization techniques, and provides interactive visualizations for portfolio performance.

🚀 Features

Portfolio optimization based on Modern Portfolio Theory

Interactive web interface (built with Flask)

Data handling with Pandas and NumPy

Integration with Kite Connect API (Zerodha)

Secure API key management using .env

Ready for deployment on any Python environment

🛠️ Installation
1. Clone the repository
git clone https://github.com/your-username/efficient-portfolio.git
cd efficient-portfolio

2. Create a virtual environment
python -m venv venv
source venv/bin/activate   # On Linux/Mac
venv\Scripts\activate      # On Windows

3. Install dependencies
pip install -r requirements.txt

🔑 Setup Environment Variables

Create a .env file in the root folder:

API_KEY=your_kiteconnect_api_key


👉 Never upload your .env file to GitHub!
Instead, share .env.example to show required variables.

▶️ Run the Application
python app.py


The app will start at:

http://127.0.0.1:5000

📂 Project Structure
Efficient Portfolio/
│── app.py               # Flask application
│── requirements.txt     # Project dependencies
│── README.md            # Project documentation
│── .gitignore           # Ignored files (env, venv, etc.)
│── .env.example         # Template for environment variables
│── templates/           # HTML templates
│── venv/                # Virtual environment (ignored in Git)

📜 License

This project is licensed under the MIT License – feel free to use and modify it.
