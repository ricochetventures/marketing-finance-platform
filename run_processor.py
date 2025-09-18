import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.data_processor import DataProcessor

# Process the data
processor = DataProcessor()
data = processor.load_and_process_excel()
processor.save_processed_data()

print("\n=== Successfully Processed ===")
for key in data.keys():
    print(f"✓ {key}")
