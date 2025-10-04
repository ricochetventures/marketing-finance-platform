import json
import pandas as pd
from pathlib import Path
import sys

def extract_companies_from_excel():
    """Extract companies from YOUR Excel file"""
    
    excel_path = 'data/2025_ResearchProject_v3.xlsx'
    
    if not Path(excel_path).exists():
        print(f"❌ Excel file not found: {excel_path}")
        return None
    
    try:
        # Read Excel file
        excel_file = pd.ExcelFile(excel_path)
        print(f"📊 Found sheets: {excel_file.sheet_names}")
        
        companies_set = set()
        
        # Extract from Stock_Prices sheet
        if 'Stock_Prices' in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name='Stock_Prices')
            print(f"Processing Stock_Prices sheet: {df.shape}")
            
            # Get from Company column if it exists
            if 'Company' in df.columns:
                company_col = df['Company'].dropna().unique()
                companies_set.update([str(c).strip() for c in company_col if str(c).strip() != 'nan'])
                print(f"Found {len(company_col)} companies in Company column")
        
        # Extract from Historical Ad Spend
        for sheet_name in excel_file.sheet_names:
            if 'Ad Spend' in sheet_name or 'Hist Ad Spend' in sheet_name:
                df = pd.read_excel(excel_file, sheet_name=sheet_name)
                if 'Company' in df.columns:
                    company_col = df['Company'].dropna().unique()
                    companies_set.update([str(c).strip() for c in company_col if str(c).strip() != 'nan'])
                    print(f"Found {len(company_col)} companies in {sheet_name}")
        
        # Extract from Historical AORs
        if 'Historical AORs V2' in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name='Historical AORs V2')
            if 'Company' in df.columns:
                company_col = df['Company'].dropna().unique()
                companies_set.update([str(c).strip() for c in company_col if str(c).strip() != 'nan'])
                print(f"Found {len(company_col)} companies in Historical AORs")
        
        # Extract from Historical Ad ROI
        if 'Historical Ad ROI' in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name='Historical Ad ROI')
            if 'Company' in df.columns:
                company_col = df['Company'].dropna().unique()
                companies_set.update([str(c).strip() for c in company_col if str(c).strip() != 'nan'])
                print(f"Found {len(company_col)} companies in Historical Ad ROI")
        
        # Clean and sort
        companies = sorted([c for c in companies_set if c and c != 'nan' and len(c) > 1])
        
        return companies
        
    except Exception as e:
        print(f"❌ Error reading Excel: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    # Create directory
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    
    # Extract companies
    companies = extract_companies_from_excel()
    
    if not companies:
        print("❌ No companies extracted. Check your Excel file.")
        sys.exit(1)
    
    # Save to JSON
    json_path = 'data/processed/companies.json'
    with open(json_path, 'w') as f:
        json.dump(companies, f, indent=2)
    
    print(f"\n✅ SUCCESS!")
    print(f"Created {json_path} with {len(companies)} companies")
    print(f"\nFirst 20 companies:")
    for i, company in enumerate(companies[:20], 1):
        print(f"{i:2d}. {company}")
    
    if len(companies) > 20:
        print(f"... and {len(companies) - 20} more")

if __name__ == "__main__":
    main()