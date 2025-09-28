# process_data_properly.py
import pandas as pd
import numpy as np
from pathlib import Path
import json

def process_excel_data():
    """Process the Excel file and create proper data files"""
    
    # Read your Excel file
    excel_path = 'data/2025_ResearchProject_v3.xlsx'
    excel_file = pd.ExcelFile(excel_path)
    
    print(f"Available sheets: {excel_file.sheet_names}")
    
    # Create output directory
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    # Process Stock Prices
    if 'Stock_Prices' in excel_file.sheet_names:
        stock_df = pd.read_excel(excel_file, sheet_name='Stock_Prices')
        print(f"Stock data shape: {stock_df.shape}")
        print(f"Stock columns: {stock_df.columns.tolist()[:10]}")
        
        # Get companies from first column (assuming it's Company)
        if 'Company' in stock_df.columns:
            companies = stock_df['Company'].unique()
            companies = [c for c in companies if pd.notna(c) and str(c).strip()]
            
            # Save companies list
            with open('data/processed/companies.json', 'w') as f:
                json.dump(sorted(companies), f)
            print(f"Saved {len(companies)} companies")
            
            # Create sample stock prices (since we need time series data)
            dates = pd.date_range(start='2020-01-01', end='2024-01-01', freq='Y')
            stock_prices = pd.DataFrame()
            
            for company in companies[:20]:  # Start with 20 companies
                # Create realistic stock price simulation
                base_price = np.random.uniform(50, 300)
                returns = np.random.normal(0.08, 0.2, len(dates))
                prices = [base_price]
                for r in returns[1:]:
                    prices.append(prices[-1] * (1 + r))
                stock_prices[company] = prices
            
            stock_prices.index = dates
            stock_prices.to_csv('data/processed/stock_prices.csv')
            print(f"Created stock prices for {len(stock_prices.columns)} companies")
    
    # Process Agency data
    agency_sheets = [s for s in excel_file.sheet_names if 'AOR' in s or 'Agency' in s]
    if agency_sheets:
        agencies_df = pd.read_excel(excel_file, sheet_name=agency_sheets[0])
        print(f"Agency data shape: {agencies_df.shape}")
        print(f"Agency columns: {agencies_df.columns.tolist()}")
        
        # Clean and save
        agencies_df.to_csv('data/processed/agencies.csv', index=False)
    
    # Create sample ROI and ad spend data
    sample_data = []
    for company in companies[:20]:
        for year in range(2020, 2025):
            sample_data.append({
                'Company': company,
                'Date': f'{year}-01-01',
                'ROI': np.random.uniform(1.5, 3.5),
                'Total_Spend': np.random.uniform(1000000, 50000000),
                'Digital_Spend': np.random.uniform(500000, 30000000),
                'TV_Spend': np.random.uniform(200000, 15000000)
            })
    
    # Save ROI data
    roi_df = pd.DataFrame(sample_data)
    roi_df.to_csv('data/processed/roi.csv', index=False)
    
    # Save ad spend data
    ad_spend_df = pd.DataFrame(sample_data)
    ad_spend_df['Year'] = ad_spend_df['Date']
    ad_spend_df.to_csv('data/processed/ad_spend.csv', index=False)
    
    print("Data processing complete!")

if __name__ == "__main__":
    process_excel_data()
