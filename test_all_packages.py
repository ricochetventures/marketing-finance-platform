#!/usr/bin/env python3
import sys
print(f"Python: {sys.version}\n")

packages = {
    # Core
    'pandas': 'pandas',
    'numpy': 'numpy',
    'scipy': 'scipy',
    'scikit-learn': 'sklearn',
    
    # ML
    'lightgbm': 'lightgbm',
    'xgboost': 'xgboost',
    'catboost': 'catboost',
    
    # Visualization
    'matplotlib': 'matplotlib',
    'seaborn': 'seaborn',
    'plotly': 'plotly',
    'dash': 'dash',
    
    # Web
    'streamlit': 'streamlit',
    'fastapi': 'fastapi',
    'uvicorn': 'uvicorn',
    'gradio': 'gradio',
    
    # Database
    'sqlalchemy': 'sqlalchemy',
    'psycopg2': 'psycopg2',
    'redis': 'redis',
    'pymongo': 'pymongo',
    
    # Time Series
    'statsmodels': 'statsmodels',
    'prophet': 'prophet',
    'pmdarima': 'pmdarima',
    
    # Others
    'requests': 'requests',
    'tqdm': 'tqdm',
    'joblib': 'joblib',
}

print("Package Status:")
print("-" * 40)

installed = []
failed = []

for name, import_name in packages.items():
    try:
        mod = __import__(import_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"✓ {name:20} {version}")
        installed.append(name)
    except ImportError:
        print(f"✗ {name:20} NOT INSTALLED")
        failed.append(name)

print("\n" + "=" * 40)
print(f"Installed: {len(installed)}/{len(packages)}")
if failed:
    print(f"Failed: {', '.join(failed)}")
else:
    print("All packages installed successfully! 🎉")
