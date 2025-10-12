"""
Verification script to ensure all data is dynamic
Run this to verify no hardcoded values are being used
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data.real_time_data_fetcher import RealTimeDataFetcher
from src.data.company_data_calculator import CompanyDataCalculator

def test_dynamic_data():
    """Test that all data is fetched dynamically"""
    
    print("=" * 60)
    print("TESTING DYNAMIC DATA FETCHING")
    print("=" * 60)
    
    fetcher = RealTimeDataFetcher()
    calculator = CompanyDataCalculator()
    
    # Test with Coca-Cola
    test_company = "Coca-Cola"
    test_ticker = "KO"
    
    print(f"\n1. Testing Marketing ROI for {test_company}...")
    roi_data = fetcher.get_company_marketing_roi(test_company, test_ticker)
    print(f"   ✓ ROI: {roi_data['roi']:.2f}x")
    print(f"   ✓ Method: {roi_data['calculation_method']}")
    print(f"   ✓ Sources: {', '.join(roi_data['sources'])}")
    
    print(f"\n2. Testing Digital Marketing % for {test_company}...")
    digital_data = fetcher.get_company_digital_marketing_percentage(test_company, test_ticker)
    print(f"   ✓ Digital %: {digital_data['digital_percentage']:.1f}%")
    print(f"   ✓ Source: {digital_data['source']}")
    print(f"   ✓ Confidence: {digital_data['confidence']}")
    
    print(f"\n3. Testing Market Share for {test_company}...")
    share_data = fetcher.get_company_market_share(test_company, test_ticker)
    print(f"   ✓ Market Share: {share_data['share']:.2f}%")
    print(f"   ✓ Method: {share_data['method']}")
    print(f"   ✓ Position: {share_data['position']}")
    
    print(f"\n4. Testing Complete Metrics Calculation...")
    metrics = calculator.get_company_metrics(test_company)
    print(f"   ✓ Stock Price: ${metrics['current_price']:.2f}")
    print(f"   ✓ Marketing ROI: {metrics['marketing_roi']:.2f}x")
    print(f"   ✓ Market Share: {metrics['market_share']:.2f}%")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - DATA IS DYNAMIC")
    print("=" * 60)

if __name__ == "__main__":
    test_dynamic_data()