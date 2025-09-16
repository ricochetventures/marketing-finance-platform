import sys
print(f"Python: {sys.version}\n")

packages = [
    'lightgbm', 'xgboost', 'pandas', 'numpy', 
    'streamlit', 'plotly', 'yfinance', 'sklearn'
]

for pkg in packages:
    try:
        mod = __import__(pkg)
        print(f"✓ {pkg} installed")
    except ImportError:
        print(f"✗ {pkg} not found")
