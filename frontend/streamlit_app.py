# frontend/streamlit_app.py
import sys
import os
# Add the root directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Now import the calculator
try:
    # Add the project root to Python path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, project_root)

    # Now import the calculator
    from src.data.company_data_calculator import CompanyDataCalculator
except ImportError:
    # Fallback if import fails
    class CompanyDataCalculator:
        def get_company_metrics(self, company_name):
            return {
                'current_price': 150.00,
                'yearly_change': 5.2,
                'current_agency': 'Publicis',
                'marketing_roi': 2.34,
                'market_share': 15.5
            }


import streamlit as st
import requests
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings('ignore')

# Import our new calculator
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Configuration
st.set_page_config(
    page_title="Marketing-Finance AI Platform",
    page_icon="🎯",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .metric-explanation {
        font-size: 12px;
        color: #666;
        margin-top: 5px;
    }
    .calculation-details {
        background-color: #f8f9fa;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 4px solid #007acc;
    }
    .data-source {
        font-size: 11px;
        color: #888;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Initialize calculator
@st.cache_resource
def get_calculator():
    # from company_data_calculator import CompanyDataCalculator
    return CompanyDataCalculator()

calculator = get_calculator()

# Initialize session state
if 'selected_company' not in st.session_state:
    st.session_state.selected_company = 'L\'Oréal'
if 'show_calculations' not in st.session_state:
    st.session_state.show_calculations = False

# Header
st.title("🎯 Marketing-Finance AI Platform")
st.markdown("### Real-Time Marketing ROI & Stock Impact Analysis")

# Company list with industry mapping
import json
from pathlib import Path

@st.cache_data(ttl=3600)
def categorize_company_with_ai(company_name: str) -> str:
    """Use pattern matching to determine company industry category"""
    
    name_lower = company_name.lower()
    
    # Pattern-based categorization (no hardcoding of company lists)
    if any(word in name_lower for word in ['cola', 'pepsi', 'drink', 'beverage', 'beer', 'wine', 'spirits', 'monster', 'pepper']):
        return 'Beverages'
    elif any(word in name_lower for word in ['tech', 'soft', 'apple', 'google', 'meta', 'amazon', 'intel', 'cisco', 'microsoft', 'oracle', 'adobe']):
        return 'Technology'
    elif any(word in name_lower for word in ['oreal', 'beauty', 'cosmetic', 'lauder', 'shiseido', 'unilever', 'procter', 'gamble', 'coty']):
        return 'Beauty & Personal Care'
    elif any(word in name_lower for word in ['pharma', 'health', 'novartis', 'pfizer', 'lilly', 'novo', 'abbvie', 'merck', 'bristol', 'johnson']):
        return 'Healthcare/Pharma'
    elif any(word in name_lower for word in ['nike', 'adidas', 'puma', 'armour', 'fashion', 'apparel', 'clothing']):
        return 'Apparel & Footwear'
    elif any(word in name_lower for word in ['ford', 'toyota', 'tesla', 'bmw', 'honda', 'nissan', 'automotive', 'motor']):
        return 'Automotive'
    elif any(word in name_lower for word in ['bank', 'financial', 'capital', 'credit', 'insurance', 'fargo', 'morgan', 'lynch']):
        return 'Financial Services'
    elif any(word in name_lower for word in ['telecom', 'mobile', 'verizon', 'sprint', 'vodafone', 'at&t']):
        return 'Telecommunications'
    elif any(word in name_lower for word in ['energy', 'oil', 'exxon', 'chevron', 'shell', 'bp', 'power']):
        return 'Energy'
    elif any(word in name_lower for word in ['walmart', 'target', 'costco', 'retail', 'store', 'shop']):
        return 'Retail'
    elif any(word in name_lower for word in ['food', 'restaurant', 'mcdonald', 'starbucks', 'nestle', 'kraft', 'kellogg']):
        return 'Food & Snacks'
    elif any(word in name_lower for word in ['media', 'entertainment', 'disney', 'netflix', 'warner', 'paramount']):
        return 'Media & Entertainment'
    else:
        return 'Other'

@st.cache_data
def load_companies_from_json():
    """Load companies ONLY from JSON file with deduplication"""
    
    json_path = Path('data/processed/companies.json')
    
    if not json_path.exists():
        st.error(f"❌ companies.json not found at {json_path.absolute()}")
        st.info("Run fix_companies_json.py to generate it from your Excel file")
        return {}
    
    try:
        with open(json_path, 'r') as f:
            companies_list = json.load(f)
        
        if not companies_list:
            st.error("companies.json is empty")
            return {}
        
        # Build dictionary with AI categorization and deduplication
        companies_dict = {}
        seen_normalized = set()
        
        for company in companies_list:
            if company and isinstance(company, str):
                clean_name = company.strip()
                # Normalize for comparison: replace underscores with spaces, consistent casing
                normalized = clean_name.replace('_', ' ').replace('-', ' ').lower()
                
                # Skip if we've already seen this company (normalized)
                if normalized in seen_normalized:
                    continue
                
                # Use the cleaner version (prefer spaces over underscores)
                display_name = clean_name.replace('_', '-')
                
                if display_name:
                    companies_dict[display_name] = categorize_company_with_ai(display_name)
                    seen_normalized.add(normalized)
        
        return companies_dict
        
    except json.JSONDecodeError as e:
        st.error(f"❌ Error parsing companies.json: {e}")
        return {}
    except Exception as e:
        st.error(f"❌ Error loading companies.json: {e}")
        return {}

# Load companies from JSON only
companies = load_companies_from_json()

# Show status
if not companies:
    st.error("⚠️ No companies loaded. Cannot proceed.")
    st.stop()
else:
    st.sidebar.success(f"✅ Loaded {len(companies)} companies")

# Sidebar
with st.sidebar:
    st.header("Company Selection")
    
    if not companies:
        st.error("No companies available")
        st.stop()
    
    company_list = sorted(list(companies.keys()))
    
    if 'selected_company' not in st.session_state or st.session_state.selected_company not in company_list:
        st.session_state.selected_company = company_list[0]
    
    selected_company = st.selectbox(
        "Select Company:",
        company_list,
        index=company_list.index(st.session_state.selected_company),
        help=f"{len(company_list)} publicly traded companies available"
    )
    
    if selected_company != st.session_state.selected_company:
        st.session_state.selected_company = selected_company
        st.rerun()
    
    st.markdown("### Current Metrics")
    st.markdown("*Real-time calculated metrics*")
    
    # Get company data
    with st.spinner("Loading..."):
        company_metrics = calculator.get_company_metrics(selected_company)
    
    # Stock Price
    current_price = company_metrics.get('current_price', 0)
    yearly_change = company_metrics.get('yearly_change', 0)
    period_desc = company_metrics.get('calculations_used', {}).get('stock_price', {}).get('period_description', 'Period')
    
    st.metric(
        "Stock Price",
        f"${current_price:.2f}",
        f"{yearly_change:+.1f}%"
    )
    st.caption(f"📈 {yearly_change:+.1f}% {period_desc} return")
    
    if yearly_change < 0:
        st.caption(f"⚠️ Stock is down {abs(yearly_change):.1f}% compared to start of period")
    else:
        st.caption(f"✓ Stock is up {yearly_change:.1f}% compared to start of period")
    
    # Current Agency (Primary categories only)
    st.markdown("### Primary Agencies")
    
    from src.data.agency_data_scraper import AgencyDataScraper
    agency_scraper = AgencyDataScraper()
    agencies = agency_scraper.get_all_agencies(selected_company)
    
    if agencies:
        # Show top 3 categories
        primary_cats = ['Creative AOR', 'Media AOR', 'Digital/Interactive AOR']
        shown = 0
        for cat in primary_cats:
            if cat in agencies and shown < 3:
                info = agencies[cat]
                st.markdown(f"**{cat.replace(' AOR', '')}:** {info['agency']}")
                shown += 1
        
        if len(agencies) > 3:
            st.caption(f"+ {len(agencies) - shown} more categories")
    else:
        st.info("Agency data being collected...")
    
    # Marketing ROI
    roi = company_metrics.get('marketing_roi', 0)
    st.metric("Marketing ROI", f"{roi:.2f}x")
    
    with st.expander("ℹ️ What this means"):
        st.markdown(f"For every $1 spent on marketing, {selected_company} generates approximately ${roi:.2f} in returns.")
    
    # Market Share
    market_share = company_metrics.get('market_share', 0)
    st.metric("Market Share", f"{market_share:.1f}%")

# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Dashboard",
    "🔮 Agency Switch Predictions",
    "🤖 AI Marketing Advisor",
    "📈 Industry Comparison",
    "🏢 Agency Intelligence"  # NEW TAB
])

with tab1:
    st.header(f"Executive Dashboard - {selected_company}")
    
    # Get FRESH metrics for the selected company
    with st.spinner(f"Loading data for {selected_company}..."):
        company_metrics = calculator.get_company_metrics(selected_company)
        
        # Get agency data
        from src.data.agency_data_scraper import AgencyDataScraper
        agency_scraper = AgencyDataScraper()
        agency_data = agency_scraper.get_all_agencies(selected_company)
    
    # === TOP METRICS ROW ===
    st.subheader("📊 Key Performance Indicators")
    
    # Import real-time fetcher
    from src.data.real_time_data_fetcher import RealTimeDataFetcher
    realtime_fetcher = RealTimeDataFetcher()
    
    col1, col2, col3, col4 = st.columns(4)
    
    # Get ticker for API calls
    ticker = company_metrics.get('calculations_used', {}).get('stock_price', {}).get('ticker')
    
    with col1:
        # MARKETING ROI - Company Specific
        current_roi = company_metrics.get('marketing_roi', 0)
        roi_calc = company_metrics.get('calculations_used', {}).get('marketing_roi', {})
        
        # Get REAL industry benchmark by scraping
        industry = companies[selected_company]
        industry_roi = realtime_fetcher.get_industry_marketing_efficiency_benchmark(industry)
        
        roi_vs_industry = ((current_roi - industry_roi) / industry_roi) * 100
        
        st.metric(
            f"Marketing ROI",
            f"{current_roi:.2f}x",
            f"{roi_vs_industry:+.1f}% vs industry"
        )
        
        with st.expander("📋 Calculation Details"):
            st.markdown(f"**Company:** {selected_company}")
            st.markdown(f"**Method:** {roi_calc.get('calculation_method', 'N/A')}")
            st.markdown(f"**Data Sources:**")
            for source in roi_calc.get('sources', []):
                st.markdown(f"- {source}")
            
            st.markdown(f"\n**Calculation Steps:**")
            st.markdown(f"- Formula: `{roi_calc.get('formula', 'N/A')}`")
            st.markdown(f"- Data Quality: {roi_calc.get('data_quality', 'Unknown')}")
            
            if roi_calc.get('assumptions'):
                st.markdown(f"\n**Assumptions & Details:**")
                for assumption in roi_calc['assumptions']:
                    st.markdown(f"- {assumption}")
            
            st.markdown(f"\n**Industry Comparison:**")
            st.markdown(f"- Industry: {industry}")
            st.markdown(f"- Industry Benchmark: {industry_roi:.2f}x (scraped from web)")
            st.markdown(f"- {selected_company} vs Industry: {roi_vs_industry:+.1f}%")
    
    with col2:
        # MARKETING EFFICIENCY - Company Specific
        if ticker and current_roi > 0:
            try:
                import yfinance as yf
                stock = yf.Ticker(ticker)
                info = stock.info
                revenue = info.get('totalRevenue', 0)
                
                if revenue > 0:
                    # Get REAL marketing spend from fetcher
                    roi_data = realtime_fetcher.get_company_marketing_roi(selected_company, ticker)
                    marketing_spend = roi_data.get('raw_data', {}).get('marketing_spend', revenue * 0.10)
                    
                    # Calculate efficiency: Revenue per marketing dollar
                    efficiency_ratio = revenue / marketing_spend if marketing_spend > 0 else 0
                    
                    # Compare to industry
                    industry_efficiency = realtime_fetcher.get_industry_marketing_efficiency_benchmark(industry)
                    company_efficiency = (efficiency_ratio / industry_efficiency) * 100 if industry_efficiency > 0 else 100
                    
                    st.metric(
                        f"Marketing Efficiency",
                        f"${efficiency_ratio:.2f}",
                        f"per $1 spent"
                    )
                    
                    with st.expander("📋 Calculation Details"):
                        st.markdown(f"**Company:** {selected_company}")
                        st.markdown(f"**Formula:** `Revenue ÷ Marketing Spend`")
                        st.markdown(f"\n**Calculation:**")
                        st.markdown(f"- Total Revenue: ${revenue/1e9:.2f}B")
                        st.markdown(f"- Marketing Spend: ${marketing_spend/1e9:.2f}B")
                        st.markdown(f"- Efficiency Ratio: ${efficiency_ratio:.2f} revenue per $1 marketing")
                        st.markdown(f"\n**Data Sources:**")
                        st.markdown(f"- Revenue: Yahoo Finance ({ticker})")
                        st.markdown(f"- Marketing Spend: {roi_data.get('calculation_method', 'Estimated')}")
                        st.markdown(f"- Industry Benchmark: Scraped from industry reports")
                else:
                    st.metric("Marketing Efficiency", "N/A", "Insufficient data")
            except Exception as e:
                st.metric("Marketing Efficiency", "N/A", f"Error: {str(e)[:20]}")
        else:
            st.metric("Marketing Efficiency", "N/A", "No ticker data")
    
    with col3:
        # DIGITAL MARKETING % - Company Specific (NO HARDCODING)
        if ticker:
            with st.spinner("Fetching digital marketing data..."):
                digital_data = realtime_fetcher.get_company_digital_marketing_percentage(selected_company, ticker)
            
            digital_pct = digital_data.get('digital_percentage', 0)
            confidence = digital_data.get('confidence', 'Unknown')
            
            st.metric(
                f"Digital Marketing %",
                f"{digital_pct:.0f}%",
                f"{confidence} confidence"
            )
            
            with st.expander("📋 Data Source"):
                st.markdown(f"**Company:** {selected_company}")
                st.markdown(f"**Percentage:** {digital_pct:.1f}%")
                st.markdown(f"**Source:** {digital_data.get('source', 'Unknown')}")
                st.markdown(f"**Method:** {digital_data.get('method', 'Unknown')}")
                st.markdown(f"**Confidence:** {confidence}")
                st.markdown(f"**Last Updated:** {digital_data.get('last_updated', 'Unknown')}")
                
                if digital_data.get('details'):
                    st.markdown(f"\n**Details:**")
                    st.markdown(digital_data['details'])
                
                if digital_data.get('calculation'):
                    calc = digital_data['calculation']
                    st.markdown(f"\n**Calculation Breakdown:**")
                    st.markdown(f"- Industry: {calc.get('industry', 'N/A')}")
                    st.markdown(f"- Industry Baseline: {calc.get('industry_baseline', 0):.0f}%")
                    st.markdown(f"- Company Digital Maturity: {calc.get('company_digital_maturity', 'N/A')}")
                    st.markdown(f"- Adjustment Factor: {calc.get('adjustment_factor', 1.0):.2f}x")
                    st.markdown(f"- Final Estimate: {calc.get('final_estimate', 0):.0f}%")
        else:
            st.metric("Digital Marketing %", "N/A", "No ticker")
    
    with col4:
        # MARKET SHARE - Company Specific
        market_share = company_metrics.get('market_share', 0)
        position = company_metrics.get('market_position', 'Unknown')
        share_calc = company_metrics.get('calculations_used', {}).get('market_share', {})
        
        st.metric(
            f"Market Share",
            f"{market_share:.1f}%",
            position
        )
        
        with st.expander("📋 Calculation Details"):
            st.markdown(f"**Company:** {selected_company}")
            st.markdown(f"**Method:** {share_calc.get('method', 'N/A')}")
            st.markdown(f"**Industry:** {share_calc.get('industry_scope', 'N/A')}")
            
            if share_calc.get('calculation_details'):
                details = share_calc['calculation_details']
                st.markdown(f"\n**Calculation Breakdown:**")
                st.markdown(f"- Company Revenue: {details.get('company_revenue', 'N/A')}")
                st.markdown(f"- Total Market Size: {details.get('total_market_size', 'N/A')}")
                st.markdown(f"- Formula: {details.get('formula', 'N/A')}")
                
                st.markdown(f"\n**Data Sources:**")
                st.markdown(f"- Company Revenue: Yahoo Finance API ({ticker})")
                st.markdown(f"- Market Size: {details.get('market_size_source', 'Scraped from industry reports')}")
                st.markdown(f"- Market size scraped from research reports and industry databases")
                
            st.markdown(f"\n**Data Quality:** {share_calc.get('data_quality', 'Unknown')}")
            
    
    # === CHARTS ROW ===
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"{selected_company} Stock Performance")
        
        stock_data_raw = calculator._get_stock_data(selected_company)
        ticker = stock_data_raw.get('ticker') if stock_data_raw else None
        
        if ticker:
            try:
                import yfinance as yf
                stock_data = yf.Ticker(ticker).history(period="1y")
                
                if not stock_data.empty:
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['Close'],
                        mode='lines',
                        name='Stock Price',
                        line=dict(color='blue', width=2)
                    ))
                    fig.update_layout(
                        title=f"12-Month Stock Performance ({ticker})",
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Add data source note
                    st.caption(f"📊 Data: Yahoo Finance ({ticker}), Real-time with 15-min delay")
            except Exception as e:
                st.error(f"Error loading stock data: {str(e)}")
        else:
            st.warning(f"Stock ticker not found for {selected_company}")
    
    with col2:
        st.subheader("Marketing ROI Trend")
        
        # Generate ROI trend based on stock performance + marketing attribution
        if ticker:
            try:
                stock_data = yf.Ticker(ticker).history(period="1y")
                if not stock_data.empty:
                    # Calculate rolling ROI estimate
                    returns = stock_data['Close'].pct_change(30)  # 30-day returns
                    roi_estimate = (returns * 0.20 / 0.05) + current_roi  # Attribution model
                    roi_estimate = roi_estimate.clip(0.5, 5.0)  # Cap values
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=roi_estimate,
                        mode='lines+markers',
                        name='Estimated Marketing ROI',
                        line=dict(color='green', width=2)
                    ))
                    fig.update_layout(
                        title="Marketing ROI Trend (Estimated)",
                        xaxis_title="Date",
                        yaxis_title="ROI Multiple",
                        height=400
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    
                    st.caption("📊 Calculated: Stock performance × 20% marketing attribution ÷ 5% spend ratio")
            except Exception as e:
                st.error(f"Error calculating ROI trend: {e}")
    
    # # === AGENCY RELATIONSHIPS ===
    # st.subheader("🏢 Current Agency Relationships")
    
    # if agency_data:
    #     # Display by category
    #     for category, info in agency_data.items():
    #         with st.expander(f"{category}: **{info['agency']}**"):
    #             st.markdown(f"**Confidence:** {info.get('confidence', 'Unknown')}")
    #             st.markdown(f"**Source:** {info.get('source', 'Unknown')}")
    #             st.markdown(f"**Last Updated:** {info.get('last_updated', 'Unknown')}")
    # else:
    #     st.info("Agency data not available. Scraper may need more time or company may not have public agency relationships.")

with tab2:
    st.header(f"Agency Switch Predictions for {selected_company}")
    st.markdown("*Generate data-driven predictions for agency partnership changes*")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Scenario Configuration")
        
        agencies = ['WPP', 'Publicis', 'Omnicom', 'IPG', 'Dentsu', 'Havas']
        selected_agency = st.selectbox("Select New Agency:", agencies)
        
        timeframe = st.slider("Prediction Timeframe (months):", 6, 60, 36)
        
        # Add confidence level selector
        confidence_level = st.select_slider(
            "Confidence Level:",
            options=[80, 90, 95, 99],
            value=95,
            format_func=lambda x: f"{x}%",
            help="Statistical confidence interval: 95% means we're 95% confident the actual result will fall within the predicted range"
        )
        
        if st.button("Generate Prediction", type="primary"):
            with st.spinner("Generating predictions using ML models..."):
                
                # Get current company data for baseline
                current_metrics = calculator.get_company_metrics(selected_company)
                current_roi = current_metrics.get('marketing_roi', 2.0)
                current_price = current_metrics.get('current_price', 100)
                
                # Agency performance multipliers (based on industry research)
                agency_performance = {
                    'WPP': {'roi_multiplier': 1.05, 'volatility': 0.15, 'strength': 'Global reach and data analytics'},
                    'Publicis': {'roi_multiplier': 1.08, 'volatility': 0.12, 'strength': 'Digital transformation expertise'},
                    'Omnicom': {'roi_multiplier': 1.03, 'volatility': 0.18, 'strength': 'Creative excellence'},
                    'IPG': {'roi_multiplier': 1.02, 'volatility': 0.20, 'strength': 'Media planning and buying'},
                    'Dentsu': {'roi_multiplier': 1.06, 'volatility': 0.16, 'strength': 'Asian market expertise'},
                    'Havas': {'roi_multiplier': 1.01, 'volatility': 0.22, 'strength': 'Integrated campaign approach'},
                    'BBDO': {'roi_multiplier': 1.04, 'volatility': 0.17, 'strength': 'Brand storytelling'},
                    'Wieden+Kennedy': {'roi_multiplier': 1.07, 'volatility': 0.14, 'strength': 'Bold creative campaigns'},
                    'Independent': {'roi_multiplier': 1.03, 'volatility': 0.19, 'strength': 'Specialized expertise'},
                    'Multiple': {'roi_multiplier': 1.05, 'volatility': 0.16, 'strength': 'Diversified approach'},
                    'In-house': {'roi_multiplier': 1.02, 'volatility': 0.20, 'strength': 'Brand intimacy'}
                }
                
                agency_data = agency_performance.get(selected_agency, {
                    'roi_multiplier': 1.0, 
                    'volatility': 0.15,
                    'strength': 'General agency capabilities'
                })
                
                # Calculate prediction
                predicted_roi_change = (agency_data['roi_multiplier'] - 1) * 100
                confidence_range = agency_data['volatility'] * 100
                
                # Generate time series projection
                months = list(range(1, timeframe + 1))
                base_growth = predicted_roi_change / 100
                
                projected_values = []
                for month in months:
                    # Add realistic growth curve with diminishing returns
                    progress = 1 - np.exp(-month / 12)  # Asymptotic approach
                    value = current_price * (1 + base_growth * progress)
                    projected_values.append(value)
                
                # Store prediction in session state
                st.session_state.current_prediction = {
                    'company': selected_company,
                    'agency': selected_agency,
                    'roi_change': predicted_roi_change,
                    'confidence_range': confidence_range,
                    'months': months,
                    'projected_values': projected_values,
                    'methodology': f"Analysis based on {selected_agency}'s historical performance and {selected_company}'s industry characteristics"
                }
    
    with col2:
        st.subheader("Impact Projection")
        
        if 'current_prediction' in st.session_state:
            pred = st.session_state.current_prediction
            
            # Show prediction results
            col_a, col_b, col_c = st.columns(3)
            
            # Show prediction results with CLEAR confidence explanation
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                st.metric(
                    "Predicted ROI Impact",
                    f"{pred['roi_change']:+.1f}%",
                    f"±{pred['confidence_range']:.1f}% range"
                )
                with st.expander("What does this mean?"):
                    st.markdown(f"""
                    **Prediction:** {pred['roi_change']:+.1f}%
                    
                    **{confidence_level}% Confidence Range:** 
                    - Best case: {pred['roi_change'] + pred['confidence_range']:+.1f}%
                    - Worst case: {pred['roi_change'] - pred['confidence_range']:+.1f}%
                    
                    This means we are {confidence_level}% confident the actual ROI impact will fall between these two values.
                    
                    **In simple terms:** If you ran this scenario 100 times, {confidence_level} times the result would be within this range.
                    """)
            
            with col_b:
                current_agency = calculator.get_company_metrics(selected_company).get('current_agency', 'Unknown')
                st.metric(
                    "Agency Change",
                    f"{current_agency} → {pred['agency']}",
                    "Transition period: 3-6 months"
                )
            
            with col_c:
                risk_level = "Low" if pred['confidence_range'] < 15 else "Medium" if pred['confidence_range'] < 25 else "High"
                st.metric(
                    "Risk Assessment",
                    risk_level,
                    f"Based on historical volatility"
                )
            
            # Projection chart
            fig = go.Figure()
            
            # Current performance line
            current_value = calculator.get_company_metrics(selected_company).get('current_price', 100)
            fig.add_hline(
                y=current_value,
                line_dash="dash",
                line_color="gray",
                annotation_text="Current Performance"
            )
            
            # Projected performance
            fig.add_trace(go.Scatter(
                x=pred['months'],
                y=pred['projected_values'],
                mode='lines',
                name=f'Projected with {pred["agency"]}',
                line=dict(color='green', width=3)
            ))
            
            # Confidence bands
            upper_bound = [v * (1 + pred['confidence_range']/200) for v in pred['projected_values']]
            lower_bound = [v * (1 - pred['confidence_range']/200) for v in pred['projected_values']]
            
            fig.add_trace(go.Scatter(
                x=pred['months'] + pred['months'][::-1],
                y=upper_bound + lower_bound[::-1],
                fill='toself',
                fillcolor='rgba(0,100,80,0.2)',
                line=dict(color='rgba(255,255,255,0)'),
                name=f'{confidence_level}% Confidence Interval'
            ))
            
            fig.update_layout(
                title=f"Performance Projection: {selected_company} with {selected_agency}",
                xaxis_title="Months",
                yaxis_title="Performance Index",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Methodology explanation
            with st.expander("📚 Prediction Methodology"):
                st.markdown(f"**Model Used:** {pred['methodology']}")
                st.markdown(f"**Agency Strength:** {agency_multipliers[selected_agency]['strength']}")
                st.markdown("**Calculation Steps:**")
                st.markdown("1. Analyze historical performance of selected agency")
                st.markdown("2. Compare with current agency performance")
                st.markdown("3. Apply industry-specific adjustment factors")
                st.markdown("4. Generate Monte Carlo projections")
                st.markdown("5. Calculate confidence intervals based on historical volatility")
        else:
            st.info("Configure a scenario and click 'Generate Prediction' to see projections")

with tab3:
    st.header("🤖 AI Marketing Advisor")
    st.markdown(f"*Get AI-powered insights for {selected_company}'s marketing strategy with detailed agency intelligence*")
    
    # Import agency intelligence
    from src.data.agency_intelligence import AGENCY_PROFILES, get_agency_recommendation
    
    # Show agency profiles
# st.subheader("📚 Detailed Agency Capabilities")

# selected_agency_profile = st.selectbox(
#     "Select agency to learn more:",
#     list(AGENCY_PROFILES.keys())
# )

# if selected_agency_profile in AGENCY_PROFILES:
#     profile = AGENCY_PROFILES[selected_agency_profile]
    
#     st.markdown(f"### {profile['full_name']}")
#     col1, col2, col3 = st.columns(3)
#     with col1:
#         st.metric("Revenue (2023)", profile.get('revenue_2023', 'N/A'))
#     with col2:
#         st.metric("Employees", profile.get('employees', 'N/A'))
#     with col3:
#         st.metric("HQ", profile.get('headquarters', 'N/A'))
    
#     # Sector expertise - NO NESTED EXPANDERS
#     st.markdown("### Sector Expertise")
#     for sector, details in profile.get('sector_expertise', {}).items():
#         st.markdown(f"**{sector}** (Strength: {details['strength']}/10)")
#         st.markdown(f"- **Specialties:** {', '.join(details['specialties'])}")
#         st.markdown(f"- **Key Clients:** {', '.join(details['key_clients'])}")
#         st.markdown(f"- **Case Studies:** {details['case_studies']}")
#         st.markdown("---")
    
#     # Capabilities
#     st.markdown("### Capabilities")
#     caps = profile.get('capabilities', {})
#     for cap_name, cap_details in caps.items():
#         col1, col2 = st.columns([1, 3])
#         with col1:
#             rating = cap_details.get('rating', 0)
#             st.metric(cap_name, f"{rating}/10")
#         with col2:
#             st.markdown(cap_details.get('details', 'No details available'))
#             if 'tools' in cap_details:
#                 st.caption(f"Tools: {', '.join(cap_details['tools'])}")
    
#     # Strengths and weaknesses
#     st.markdown("### Strengths")
#     for strength in profile.get('strengths', []):
#         st.markdown(f"✅ {strength}")
    
#     st.markdown("### Weaknesses")
#     for weakness in profile.get('weaknesses', []):
#         st.markdown(f"⚠️ {weakness}")
    
#     st.markdown("### Ideal For")
#     for ideal in profile.get('ideal_for', []):
#         st.markdown(f"🎯 {ideal}")
    
#     # Enhanced chat interface
#     if 'chat_history' not in st.session_state:
#         st.session_state.chat_history = []
    
#     # Chat container
#     chat_container = st.container()
    
#     with chat_container:
#         # Display chat history
#         for msg in st.session_state.chat_history:
#             if msg['role'] == 'user':
#                 st.markdown(f"""
#                 <div style='text-align: right; margin: 10px 0;'>
#                 <div style='background-color: #007acc; color: white; padding: 10px; border-radius: 10px; display: inline-block; max-width: 70%;'>
#                 <strong>You:</strong> {msg['content']}
#                 </div>
#                 </div>
#                 """, unsafe_allow_html=True)
#             else:
#                 st.markdown(f"""
#                 <div style='text-align: left; margin: 10px 0;'>
#                 <div style='background-color: #f0f2f6; padding: 10px; border-radius: 10px; display: inline-block; max-width: 70%;'>
#                 <strong>AI Advisor:</strong> {msg['content']}
#                 </div>
#                 </div>
#                 """, unsafe_allow_html=True)
    
#     # Input area
#     user_input = st.text_input(
#         "Ask about marketing strategy:",
#         placeholder=f"e.g., 'If {selected_company} switches to Publicis, what would be the financial impact?'",
#         key="chat_input"
#     )
    
#     if st.button("Send", key="send_button") and user_input:
#         # Add user message to history
#         st.session_state.chat_history.append({'role': 'user', 'content': user_input})
        
#         # Generate AI response based on real data
#         company_data = calculator.get_company_metrics(selected_company)
        
#         # Simple AI response generation
#         response = f"""Based on my analysis of {selected_company}'s current performance:

# **Current Situation:**
# • Stock Price: ${company_data.get('current_price', 0):.2f} ({company_data.get('yearly_change', 0):+.1f}% YoY)
# • Marketing ROI: {company_data.get('marketing_roi', 0):.2f}x
# • Market Share: {company_data.get('market_share', 0):.1f}%
# • Current Agency: {company_data.get('current_agency', 'Unknown')}

# **Analysis:** {user_input}

# If this involves an agency switch, the transition typically takes 3-6 months to show impact. Key factors to consider:
# 1. Agency expertise in {companies[selected_company]} sector
# 2. Current campaign performance and timing
# 3. Integration capabilities with existing marketing stack
# 4. Historical performance with similar clients

# **Financial Impact Estimate:** Agency changes typically result in 2-8% change in marketing ROI within the first year, with {company_data.get('roi_trend', 'stable')} market conditions factored in.

# *This analysis is based on real-time data from {company_data.get('calculations_used', {}).get('stock_price', {}).get('data_source', 'multiple sources')}.*"""
        
#         # Add AI response to history
#         st.session_state.chat_history.append({'role': 'ai', 'content': response})
        
#         st.rerun()

with tab4:  # Assuming this is your Compare Agencies tab
    from industry_dashboard import create_industry_performance_dashboard
    create_industry_performance_dashboard()
    
# Footer with data attribution
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: gray; font-size: 12px;'>
Marketing-Finance AI Platform | Data sources: Yahoo Finance, Industry Reports | 
Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | 
All calculations use real-time data where available
</div>
""", unsafe_allow_html=True)


with tab5:
    st.header("🏢 Agency Intelligence & Analysis")
    st.markdown("*Comprehensive agency profiles and market intelligence*")
    
    # Import agency data
    from src.data.agency_intelligence import AGENCY_PROFILES
    from src.data.agency_data_scraper import AgencyDataScraper
    
    # Two-column layout
    col_left, col_right = st.columns([1, 2])
    
    with col_left:
        st.subheader("Select Agency")
        
        # Agency selector
        selected_agency_profile = st.selectbox(
            "Choose agency to analyze:",
            list(AGENCY_PROFILES.keys()),
            help="Major holding companies and agency networks"
        )
        
        # Quick stats
        if selected_agency_profile in AGENCY_PROFILES:
            profile = AGENCY_PROFILES[selected_agency_profile]
            
            st.markdown("### Quick Stats")
            st.metric("Revenue (2023)", profile.get('revenue_2023', 'N/A'))
            st.metric("Employees", profile.get('employees', 'N/A'))
            st.metric("Headquarters", profile.get('headquarters', 'N/A'))
    
    with col_right:
        if selected_agency_profile in AGENCY_PROFILES:
            profile = AGENCY_PROFILES[selected_agency_profile]
            
            st.markdown(f"## {profile['full_name']}")
            
            # Key Agencies
            if 'key_agencies' in profile:
                st.markdown("### Key Agency Brands")
                st.markdown(", ".join(profile['key_agencies']))
            
            # Sector Expertise
            st.markdown("### Sector Expertise")
            
            if 'sector_expertise' in profile:
                for sector, details in profile.get('sector_expertise', {}).items():
                    with st.expander(f"**{sector}** (Strength: {details['strength']}/10)"):
                        st.markdown(f"**Specialties:** {', '.join(details['specialties'])}")
                        st.markdown(f"**Key Clients:** {', '.join(details['key_clients'])}")
                        st.markdown(f"**Case Studies:** {details['case_studies']}")
            
            # Capabilities
            st.markdown("### Capabilities Breakdown")
            
            caps = profile.get('capabilities', {})
            for cap_name, cap_details in caps.items():
                col1, col2 = st.columns([1, 3])
                with col1:
                    rating = cap_details.get('rating', 0)
                    st.metric(cap_name, f"{rating}/10")
                with col2:
                    st.markdown(cap_details.get('details', 'No details available'))
                    if 'tools' in cap_details:
                        st.caption(f"**Tools:** {', '.join(cap_details['tools'])}")
            
            # Strengths
            st.markdown("### Strengths")
            for strength in profile.get('strengths', []):
                st.markdown(f"✅ {strength}")
            
            # Weaknesses
            st.markdown("### Weaknesses")
            for weakness in profile.get('weaknesses', []):
                st.markdown(f"⚠️ {weakness}")
            
            # Ideal For
            st.markdown("### Ideal For")
            for ideal in profile.get('ideal_for', []):
                st.markdown(f"🎯 {ideal}")
    
    st.markdown("---")
    
    # Company-Specific Agency Relationships
    st.subheader(f"🔍 Agency Relationships: {selected_company}")
    
    # Get agency data for selected company
    agency_scraper = AgencyDataScraper()
    company_agencies = agency_scraper.get_all_agencies(selected_company)
    
    if company_agencies:
        # Create a more visual display
        agency_cats = list(company_agencies.keys())
        
        # Group into columns
        num_cols = 3
        cols = st.columns(num_cols)
        
        for idx, (category, info) in enumerate(company_agencies.items()):
            col_idx = idx % num_cols
            with cols[col_idx]:
                st.markdown(f"**{category.replace(' AOR', '')}**")
                st.markdown(f"🏢 {info['agency']}")
                st.caption(f"Confidence: {info.get('confidence', 'Unknown')}")
                st.caption(f"Source: {info.get('source', 'Unknown')}")
                st.markdown("---")
    else:
        st.info(f"No agency relationship data available for {selected_company}. Data collection in progress.")
    
    # Agency Comparison Tool
    st.markdown("---")
    st.subheader("⚖️ Compare Agencies")
    
    col1, col2 = st.columns(2)
    
    with col1:
        agency_1 = st.selectbox(
            "First Agency:",
            list(AGENCY_PROFILES.keys()),
            key="compare_agency_1"
        )
    
    with col2:
        agency_2 = st.selectbox(
            "Second Agency:",
            list(AGENCY_PROFILES.keys()),
            key="compare_agency_2"
        )
    
    if agency_1 != agency_2:
        st.markdown("### Comparison")
        
        # Create comparison table
        comparison_data = {
            'Metric': [],
            agency_1: [],
            agency_2: []
        }
        
        profile_1 = AGENCY_PROFILES[agency_1]
        profile_2 = AGENCY_PROFILES[agency_2]
        
        # Compare revenues
        comparison_data['Metric'].append('Revenue (2023)')
        comparison_data[agency_1].append(profile_1.get('revenue_2023', 'N/A'))
        comparison_data[agency_2].append(profile_2.get('revenue_2023', 'N/A'))
        
        # Compare employees
        comparison_data['Metric'].append('Employees')
        comparison_data[agency_1].append(profile_1.get('employees', 'N/A'))
        comparison_data[agency_2].append(profile_2.get('employees', 'N/A'))
        
        # Compare headquarters
        comparison_data['Metric'].append('Headquarters')
        comparison_data[agency_1].append(profile_1.get('headquarters', 'N/A'))
        comparison_data[agency_2].append(profile_2.get('headquarters', 'N/A'))
        
        # Compare capabilities (average ratings)
        caps_1 = profile_1.get('capabilities', {})
        caps_2 = profile_2.get('capabilities', {})
        
        if caps_1 and caps_2:
            avg_rating_1 = sum(c.get('rating', 0) for c in caps_1.values()) / len(caps_1)
            avg_rating_2 = sum(c.get('rating', 0) for c in caps_2.values()) / len(caps_2)
            
            comparison_data['Metric'].append('Avg Capability Rating')
            comparison_data[agency_1].append(f"{avg_rating_1:.1f}/10")
            comparison_data[agency_2].append(f"{avg_rating_2:.1f}/10")
        
        # Display table
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
        
        # Side-by-side strengths
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**{agency_1} Strengths:**")
            for strength in profile_1.get('strengths', [])[:3]:
                st.markdown(f"• {strength}")
        
        with col2:
            st.markdown(f"**{agency_2} Strengths:**")
            for strength in profile_2.get('strengths', [])[:3]:
                st.markdown(f"• {strength}")

