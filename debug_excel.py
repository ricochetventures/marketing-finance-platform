import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Read your Excel file
excel_path = 'data/2025_ResearchProject_v3.xlsx'
excel_file = pd.ExcelFile(excel_path)

print("=== EXCEL FILE STRUCTURE ===\n")
print(f"Sheet names: {excel_file.sheet_names}\n")

# Examine each sheet
for sheet_name in excel_file.sheet_names:
    print(f"\n=== Sheet: {sheet_name} ===")
    df = pd.read_excel(excel_file, sheet_name=sheet_name)
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"First few rows:\n{df.head(3)}")
    print(f"Data types:\n{df.dtypes}")
    print("-" * 50)
