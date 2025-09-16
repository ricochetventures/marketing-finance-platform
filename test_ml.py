import sys
print(f"Python version: {sys.version}")

packages = {
    'numpy': None,
    'pandas': None,
    'sklearn': None,
    'xgboost': None,
}

for package in packages:
    try:
        if package == 'sklearn':
            import sklearn
            packages[package] = sklearn.__version__
        else:
            mod = __import__(package)
            packages[package] = mod.__version__
        print(f"✓ {package}: {packages[package]}")
    except ImportError:
        print(f"✗ {package}: Not installed")

# Try LightGBM separately
try:
    import lightgbm
    print(f"✓ lightgbm: {lightgbm.__version__}")
except ImportError:
    print("✗ lightgbm: Not installed (use XGBoost as alternative)")
