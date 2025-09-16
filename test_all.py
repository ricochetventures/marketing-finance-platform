#!/usr/bin/env python3

import sys
print(f"Python: {sys.version}\n")

def test_import(module_name, display_name=None):
    """Test if a module can be imported"""
    if display_name is None:
        display_name = module_name
    try:
        mod = __import__(module_name)
        version = getattr(mod, '__version__', 'installed')
        print(f"✓ {display_name:20} {version}")
        return True
    except ImportError:
        print(f"✗ {display_name:20} NOT INSTALLED")
        return False

print("Core Libraries:")
print("-" * 40)
test_import('pandas')
test_import('numpy')
test_import('scipy')
test_import('sklearn', 'scikit-learn')

print("\nML Libraries:")
print("-" * 40)
test_import('lightgbm')
test_import('xgboost')

print("\nVisualization:")
print("-" * 40)
test_import('matplotlib')
test_import('seaborn')
test_import('plotly')

print("\nWeb Frameworks:")
print("-" * 40)
test_import('streamlit')
test_import('fastapi')

print("\nFinancial Data:")
print("-" * 40)
test_import('yfinance')

print("\nUtilities:")
print("-" * 40)
test_import('requests')
test_import('bs4', 'beautifulsoup4')
test_import('sqlalchemy')

print("\n✅ Environment setup complete!")
