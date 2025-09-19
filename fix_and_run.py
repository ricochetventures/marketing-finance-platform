import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add project to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("🔧 Fixing data structure issues...")

# Create necessary directories
Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("models/saved").mkdir(parents=True, exist_ok=True)

# Read and process the Excel file
excel_path = 'data/2025_ResearchProject_v3.xlsx'
excel_file = pd.ExcelFile(excel_path)

print(f"📊 Found sheets: {excel_file.sheet_names}")

# Process Stock Prices
if 'Stock_Prices' in excel_file.sheet_names:
    df = pd.read_excel(excel_file, sheet_name='Stock_Prices')
    
    # Get company column and year columns
    company_col = 'Company'
    year_cols = [c for c in df.columns if c != company_col]
    
    # Melt to long format
    df_long = df.melt(id_vars=[company_col], value_vars=year_cols, var_name='Year', value_name='Price')
    df_long['Date'] = pd.to_datetime(df_long['Year'].astype(str) + '-01-01', errors='coerce')
    df_long = df_long.dropna(subset=['Date', 'Price'])
    
    # Pivot for companies as columns
    stock_prices = df_long.pivot(index='Date', columns=company_col, values='Price')
    stock_prices.to_csv('data/processed/stock_prices.csv')
    print(f"✅ Processed stock prices: {stock_prices.shape}")

# Save company list
companies = list(stock_prices.columns)
import json
with open('data/processed/companies.json', 'w') as f:
    json.dump(companies, f)
    
print(f"✅ Found {len(companies)} companies")
print("🎉 Data processing complete! You can now run the application.")
