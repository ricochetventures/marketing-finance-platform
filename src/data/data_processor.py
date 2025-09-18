import pandas as pd
import numpy as np
from pathlib import Path
import json

class DataProcessor:
    def __init__(self, excel_path='data/2025_ResearchProject_v3.xlsx'):
        self.excel_path = excel_path
        self.processed_data = {}
        
    def load_and_process_excel(self):
        """Load all sheets from Excel and process them correctly"""
        
        excel_file = pd.ExcelFile(self.excel_path)
        print(f"Available sheets: {excel_file.sheet_names}")
        
        # Process Stock Prices
        if 'Stock_Prices' in excel_file.sheet_names:
            stock_prices_raw = pd.read_excel(excel_file, sheet_name='Stock_Prices')
            print(f"\nProcessing Stock Prices...")
            print(f"Raw shape: {stock_prices_raw.shape}")
            print(f"Columns: {stock_prices_raw.columns.tolist()[:10]}...")
            
            # Get company column
            company_col = 'Company'
            
            # Get year columns - those containing 'Stock_' or just years
            year_columns = []
            for col in stock_prices_raw.columns:
                if col != company_col:
                    col_str = str(col)
                    if 'Stock_' in col_str or col_str.isdigit():
                        year_columns.append(col)
            
            print(f"Found {len(year_columns)} year columns")
            
            if year_columns:
                # Melt to long format
                stock_prices_long = stock_prices_raw[[company_col] + year_columns].melt(
                    id_vars=[company_col],
                    value_vars=year_columns,
                    var_name='Year_Col',
                    value_name='Price'
                )
                
                # Extract year from column name
                stock_prices_long['Year'] = stock_prices_long['Year_Col'].str.extract(r'(\d{4})', expand=False)
                
                # Handle cases where year might be the column itself
                stock_prices_long['Year'] = stock_prices_long['Year'].fillna(
                    stock_prices_long['Year_Col'].apply(lambda x: str(x) if str(x).isdigit() else None)
                )
                
                # Drop rows without valid years
                stock_prices_long = stock_prices_long.dropna(subset=['Year'])
                
                # Convert to datetime
                stock_prices_long['Date'] = pd.to_datetime(
                    stock_prices_long['Year'].astype(str) + '-01-01'
                )
                
                # Convert price to numeric
                stock_prices_long['Price'] = pd.to_numeric(
                    stock_prices_long['Price'], errors='coerce'
                )
                
                # Remove duplicates - keep the first occurrence
                stock_prices_long = stock_prices_long.drop_duplicates(
                    subset=['Company', 'Date'], keep='first'
                )
                
                # Remove NaN prices
                stock_prices_long = stock_prices_long.dropna(subset=['Price'])
                
                print(f"After processing: {len(stock_prices_long)} records")
                print(f"Unique companies: {stock_prices_long['Company'].nunique()}")
                print(f"Date range: {stock_prices_long['Date'].min()} to {stock_prices_long['Date'].max()}")
                
                # Try to pivot - if it fails, aggregate duplicates
                try:
                    stock_prices_pivot = stock_prices_long.pivot(
                        index='Date',
                        columns='Company',
                        values='Price'
                    )
                except:
                    print("Pivot failed, aggregating duplicates...")
                    # If pivot fails, aggregate by mean
                    stock_prices_pivot = stock_prices_long.pivot_table(
                        index='Date',
                        columns='Company',
                        values='Price',
                        aggfunc='mean'  # Take mean of duplicates
                    )
                
                self.processed_data['stock_prices'] = stock_prices_pivot
                print(f"Final stock prices shape: {stock_prices_pivot.shape}")
        
        # Process Historical Ad Spend
        ad_spend_sheet = None
        for sheet in excel_file.sheet_names:
            if 'Ad Spend' in sheet or 'Hist Ad Spend' in sheet:
                ad_spend_sheet = sheet
                break
        
        if ad_spend_sheet:
            ad_spend_raw = pd.read_excel(excel_file, sheet_name=ad_spend_sheet)
            print(f"\nProcessing Ad Spend from sheet: {ad_spend_sheet}")
            
            company_col = 'Company'
            year_columns = []
            
            for col in ad_spend_raw.columns:
                try:
                    if str(col).isdigit() and 2000 <= int(col) <= 2030:
                        year_columns.append(col)
                except:
                    continue
            
            if year_columns:
                ad_spend_long = ad_spend_raw[[company_col] + year_columns].melt(
                    id_vars=[company_col],
                    value_vars=year_columns,
                    var_name='Year',
                    value_name='Total_Spend'
                )
                
                ad_spend_long['Year'] = pd.to_datetime(
                    ad_spend_long['Year'].astype(str) + '-01-01'
                )
                ad_spend_long['Total_Spend'] = pd.to_numeric(
                    ad_spend_long['Total_Spend'], errors='coerce'
                )
                
                # Remove duplicates
                ad_spend_long = ad_spend_long.drop_duplicates(
                    subset=['Company', 'Year'], keep='first'
                )
                
                ad_spend_long = ad_spend_long.dropna(subset=['Total_Spend'])
                
                # Create synthetic splits
                ad_spend_long['Digital_Spend'] = ad_spend_long['Total_Spend'] * 0.6
                ad_spend_long['TV_Spend'] = ad_spend_long['Total_Spend'] * 0.3
                ad_spend_long['Print_Spend'] = ad_spend_long['Total_Spend'] * 0.1
                
                self.processed_data['ad_spend'] = ad_spend_long
                print(f"Processed ad spend shape: {ad_spend_long.shape}")
        
        # Process Agencies
        if 'Historical AORs V2' in excel_file.sheet_names:
            agencies_raw = pd.read_excel(excel_file, sheet_name='Historical AORs V2')
            print(f"\nProcessing Agencies...")
            
            agencies = agencies_raw.copy()
            
            # Clean up column names
            agencies.columns = agencies.columns.str.strip()
            
            # Ensure we have Company and Agency columns
            if 'AOR' in agencies.columns and 'Agency' not in agencies.columns:
                agencies['Agency'] = agencies['AOR']
            
            # Remove duplicates
            if 'Company' in agencies.columns and 'Agency' in agencies.columns:
                agencies = agencies.drop_duplicates(
                    subset=['Company', 'Agency'], keep='first'
                )
            
            # Add a year column if exists
            if 'Year' in agencies.columns:
                agencies['Start_Date'] = pd.to_datetime(
                    agencies['Year'].astype(str) + '-01-01', errors='coerce'
                )
            else:
                agencies['Start_Date'] = pd.Timestamp.now()
            
            self.processed_data['agencies'] = agencies
            print(f"Processed agencies shape: {agencies.shape}")
        
        # Process ROI
        if 'Historical Ad ROI' in excel_file.sheet_names:
            roi_raw = pd.read_excel(excel_file, sheet_name='Historical Ad ROI')
            print(f"\nProcessing ROI...")
            
            company_col = 'Company'
            year_columns = []
            
            for col in roi_raw.columns:
                try:
                    if str(col).isdigit() and 2000 <= int(col) <= 2030:
                        year_columns.append(col)
                except:
                    continue
            
            if year_columns:
                roi_long = roi_raw[[company_col] + year_columns].melt(
                    id_vars=[company_col],
                    value_vars=year_columns,
                    var_name='Year',
                    value_name='ROI'
                )
                
                roi_long['Date'] = pd.to_datetime(
                    roi_long['Year'].astype(str) + '-01-01'
                )
                roi_long['ROI'] = pd.to_numeric(roi_long['ROI'], errors='coerce')
                
                # Remove duplicates
                roi_long = roi_long.drop_duplicates(
                    subset=['Company', 'Date'], keep='first'
                )
                
                roi_long = roi_long.dropna(subset=['ROI'])
                
                self.processed_data['roi'] = roi_long
                print(f"Processed ROI shape: {roi_long.shape}")
        
        # Get unique companies
        companies = set()
        
        if 'stock_prices' in self.processed_data:
            companies.update(self.processed_data['stock_prices'].columns)
        
        for key in ['ad_spend', 'agencies', 'roi']:
            if key in self.processed_data and 'Company' in self.processed_data[key].columns:
                companies.update(self.processed_data[key]['Company'].unique())
        
        # Clean company names
        companies = [str(c).strip() for c in companies if pd.notna(c) and str(c).strip()]
        
        self.processed_data['companies'] = sorted(list(companies))
        print(f"\nTotal unique companies: {len(self.processed_data['companies'])}")
        if self.processed_data['companies']:
            print(f"Sample companies: {self.processed_data['companies'][:5]}")
        
        return self.processed_data
    
    def save_processed_data(self):
        """Save processed data for API use"""
        output_dir = Path('data/processed')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for key, df in self.processed_data.items():
            try:
                if isinstance(df, pd.DataFrame):
                    df.to_csv(output_dir / f'{key}.csv', index=True)
                    print(f"Saved {key}.csv")
                elif isinstance(df, list):
                    with open(output_dir / f'{key}.json', 'w') as f:
                        json.dump(df, f)
                    print(f"Saved {key}.json")
            except Exception as e:
                print(f"Error saving {key}: {e}")

# Run the processor
if __name__ == "__main__":
    try:
        processor = DataProcessor()
        data = processor.load_and_process_excel()
        processor.save_processed_data()
        
        print("\n✅ Data processing complete!")
        print("\n=== Data Summary ===")
        for key, value in data.items():
            if isinstance(value, pd.DataFrame):
                print(f"{key}: {value.shape}")
            elif isinstance(value, list):
                print(f"{key}: {len(value)} items")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
