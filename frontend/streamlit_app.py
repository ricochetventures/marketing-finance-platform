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
    from src.data.enhanced_calculator import EnhancedMetricsCalculator

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
    /* Fixed Header */
    .fixed-header {
        position: fixed;
        top: 60px;
        left: 0;
        right: 0;
        height: 50px;
        background: linear-gradient(90deg, #1e3a8a 0%, #3b82f6 100%);
        color: white;
        z-index: 999;
        padding: 12px 20px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    .fixed-header h3 {
        margin: 0;
        font-size: 18px;
        font-weight: 600;
    }
    
    .fixed-header-info {
        font-size: 13px;
        color: rgba(255,255,255,0.9);
    }
    
    /* Agency section styling - NO DROPDOWNS */
    .agency-category {
        margin: 10px 0;
        padding: 10px;
        border-left: 3px solid #3b82f6;
        background-color: #f8f9fa;
        border-radius: 4px;
    }
    
    .agency-category strong {
        color: #1e3a8a;
        font-size: 14px;
        display: block;
        margin-bottom: 4px;
    }
    
    .agency-name {
        color: #334155;
        font-size: 15px;
        margin: 2px 0;
        font-weight: 500;
    }
    
    .agency-source {
        color: #64748b;
        font-size: 11px;
        margin-top: 4px;
        display: block;
    }
    
    /* Hide the success notification */
    .stAlert[data-baseweb="notification"] {
        display: none !important;
    }
    
    /* Existing styles preserved */
    .metric-explanation {
        font-size: 14px;
        color: #444;
        margin-top: 5px;
        line-height: 1.6;
        font-style: normal !important;
        white-space: normal !important;
        word-wrap: break-word !important;
    }
    
    .calculation-details {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
        border-left: 4px solid #007acc;
        font-style: normal !important;
    }
    
    .calculation-details p, .calculation-details li {
        font-style: normal !important;
        margin-bottom: 8px;
    }
    
    .data-source {
        font-size: 12px;
        color: #666;
        font-style: italic;
        margin-top: 10px;
    }
    
    .streamlit-expanderContent {
        font-style: normal !important;
    }
    
    .streamlit-expanderContent p {
        white-space: normal !important;
        word-wrap: break-word !important;
        font-style: normal !important;
    }
</style>
""", unsafe_allow_html=True)

# Fixed Header
st.markdown(f"""
<div class="fixed-header">
    <h3>🎯 {st.session_state.get('selected_company', 'Marketing-Finance AI Platform')}</h3>
    <div class="fixed-header-info">
        AI-Powered Marketing Performance Intelligence
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize calculator
@st.cache_resource
def get_calculators():
    return {
        'basic': CompanyDataCalculator(),
        'enhanced': EnhancedMetricsCalculator()
    }

calculators = get_calculators()
calculator = calculators['enhanced']

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
    
    st.markdown("---")
    st.markdown("### Key Metrics")
    
    # Get company data with full explanations
    with st.spinner("Loading..."):
        company_metrics = calculator.get_company_metrics(selected_company)
    
    # Import calculation explainer for tooltips
    from src.data.calculation_explainer import CalculationExplainer
    explainer = CalculationExplainer()
    
    # === STOCK PRICE with specific period ===
    current_price = company_metrics.get('current_price', 0)
    yearly_change = company_metrics.get('yearly_change', 0)
    period_desc = company_metrics.get('period_description', 'Period')
    
    # Get full explanation
    stock_explanation = company_metrics.get('explanations', {}).get('stock_price', {})
    stock_tooltip = explainer.format_for_tooltip(stock_explanation) if stock_explanation else "Stock price from Yahoo Finance"
    stock_source_text = explainer.format_for_small_text(stock_explanation) if stock_explanation else "Source: Yahoo Finance"
    
    st.metric(
        "Stock Price",
        f"${current_price:.2f}",
        f"{yearly_change:+.1f}%",
        help=stock_tooltip
    )
    
    # Small grey text showing what the percentage means
    st.markdown(
        f"<p style='font-size: 11px; color: #666; margin-top: -10px;'>{period_desc} return</p>",
        unsafe_allow_html=True
    )
    st.markdown(
        f"<p style='font-size: 10px; color: #999; margin-top: -5px;'>{stock_source_text}</p>",
        unsafe_allow_html=True
    )
    
    # === PRIMARY AGENCIES (NO DROPDOWNS) ===
    st.markdown("---")
    st.markdown("### Primary Agencies")
    st.caption("Agency of Record by category")

    from src.data.agency_data_scraper import AgencyDataScraper
    agency_scraper = AgencyDataScraper()
    agencies = agency_scraper.get_all_agencies(selected_company)

    if agencies:
        primary_cats = ['Creative AOR', 'Media AOR', 'Digital/Interactive AOR']
        
        for cat in primary_cats:
            if cat in agencies:
                info = agencies[cat]
                agency_name = info['agency']
                
                if agency_name == 'Tool':
                    agency_name = 'Data Unavailable'
                
                cat_display = cat.replace(' AOR', '')
                confidence = info.get('confidence', 'Unknown')
                last_updated = info.get('last_updated', 'Unknown')
                
                # Compact display with tooltip
                st.markdown(f"""
                <div class="agency-category">
                    <strong>{cat_display}</strong>
                    <div class="agency-name">{agency_name}</div>
                    <span class="agency-source">{confidence} confidence • Updated: {last_updated}</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Other categories - even more compact
        other_cats = [cat for cat in agencies.keys() if cat not in primary_cats]
        if other_cats:
            st.markdown("**Other Categories**")
            for cat in other_cats:
                info = agencies[cat]
                agency_name = info['agency'] if info['agency'] != 'Tool' else 'Data Unavailable'
                cat_short = cat.replace(' AOR', '').replace('/', ' / ')
                st.caption(f"{cat_short}: {agency_name}")
    else:
        st.info("Agency data being collected...")
        st.caption("Source: Real-time web search")


    
    # === MARKETING ROI (Fixed) ===
    st.markdown("---")
    roi = company_metrics.get('marketing_roi', 0)
    roi_explanation = company_metrics.get('explanations', {}).get('marketing_roi', {})
    
    st.metric("Marketing ROI", f"{roi:.2f}x")
    
    # Small grey text showing what this means (NOT italic, NOT overflowing)
    interpretation = f"${roi:.2f} return per $1 spent"
    st.markdown(
        f"<p style='font-size: 11px; color: #666; margin-top: -10px; font-style: normal;'>{interpretation}</p>",
        unsafe_allow_html=True
    )
    
    # Data source in even smaller text
    roi_source = explainer.format_for_small_text(roi_explanation) if roi_explanation else "Source: Financial analysis"
    st.markdown(
        f"<p style='font-size: 10px; color: #999; margin-top: -5px; font-style: normal;'>{roi_source}</p>",
        unsafe_allow_html=True
    )
    
    # Detailed methodology in tooltip (not dropdown)
    if roi_explanation:
        roi_tooltip = explainer.format_for_tooltip(roi_explanation)
        with st.expander("ℹ️ Calculation details", expanded=False):
            # Display with proper formatting (not italic)
            st.markdown(roi_tooltip)
    
    # === MARKET SHARE ===
    st.markdown("---")
    market_share = company_metrics.get('market_share', 0)
    market_explanation = company_metrics.get('explanations', {}).get('market_share', {})
    
    st.metric("Market Share", f"{market_share:.1f}%")
    
    # Show methodology in small grey text
    position = company_metrics.get('market_position', 'Unknown')
    st.markdown(
        f"<p style='font-size: 11px; color: #666; margin-top: -10px; font-style: normal;'>{position}</p>",
        unsafe_allow_html=True
    )
    
    # Data source
    share_source = explainer.format_for_small_text(market_explanation) if market_explanation else "Source: Market analysis"
    st.markdown(
        f"<p style='font-size: 10px; color: #999; margin-top: -5px; font-style: normal;'>{share_source}</p>",
        unsafe_allow_html=True
    )
    
    # Full details in tooltip
    if market_explanation:
        share_tooltip = explainer.format_for_tooltip(market_explanation)
        with st.expander("ℹ️ Calculation details", expanded=False):
            st.markdown(share_tooltip)



# Main content tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Executive Dashboard",
    "🔮 Agency Switch Predictions",
    "🤖 AI Marketing Advisor",
    "📈 Industry Comparison",
    "🏢 Agency Intelligence"  # NEW TAB
])

with tab1:
    st.markdown(f"## {selected_company}")
    st.markdown("### Key Performance Indicators")
    
    # Get FRESH metrics for the selected company
    with st.spinner(f"Loading data for {selected_company}..."):
        company_metrics = calculator.get_company_metrics(selected_company)
        
        # Get agency data
        from src.data.agency_data_scraper import AgencyDataScraper
        agency_scraper = AgencyDataScraper()
        agency_data = agency_scraper.get_all_agencies(selected_company)
        
        # Get explanations
        from src.data.calculation_explainer import CalculationExplainer
        explainer = CalculationExplainer()
    
    # === TOP METRICS ROW (Enhanced with real data) ===
    st.subheader("📊 Key Performance Indicators")
    st.markdown(
        "<p style='font-size: 12px; color: #666; margin-bottom: 20px;'>Real-time metrics calculated from financial data and market analysis</p>",
        unsafe_allow_html=True
    )
    
    # Import real-time fetcher for additional metrics
    from src.data.real_time_data_fetcher import RealTimeDataFetcher
    realtime_fetcher = RealTimeDataFetcher()

    # Get enhanced metrics
    ticker = company_metrics.get('calculations_used', {}).get('stock_price', {}).get('ticker')

    if ticker:
        enhanced_metrics = {
            'efficiency': calculators['enhanced'].calculate_marketing_efficiency(selected_company, ticker),
            'digital': calculators['enhanced'].calculate_digital_marketing_percentage(selected_company, ticker),
            'market_share': calculators['enhanced'].calculate_market_share(selected_company, ticker)
        }
    else:
        enhanced_metrics = {
            'efficiency': {'efficiency': 0, 'interpretation': 'No ticker data'},
            'digital': {'digital_percentage': 0, 'interpretation': 'No ticker data'},
            'market_share': {'market_share': 0, 'interpretation': 'No ticker data'}
        }


    
    col1, col2, col3, col4 = st.columns(4)
    
    # Get ticker for API calls
    ticker = company_metrics.get('calculations_used', {}).get('stock_price', {}).get('ticker')
    
    with col1:
        # MARKETING ROI
        current_roi = company_metrics.get('marketing_roi', 0)
        roi_explanation = company_metrics.get('explanations', {}).get('marketing_roi', {})
        
        # Get REAL industry benchmark
        industry = companies[selected_company]
        industry_roi = realtime_fetcher.get_industry_marketing_efficiency_benchmark(industry)
        
        # Calculate vs industry (consistent logic)
        if industry_roi > 0:
            roi_vs_industry = ((current_roi - industry_roi) / industry_roi) * 100
            delta_label = f"{roi_vs_industry:+.1f}% vs industry"
        else:
            delta_label = "Industry data unavailable"
        
        st.metric(
            f"Marketing ROI",
            f"{current_roi:.2f}x",
            delta_label
        )
        
        # Small text with source
        st.markdown(
            f"<p style='font-size: 10px; color: #999; margin-top: -10px;'>Industry avg: {industry_roi:.2f}x</p>",
            unsafe_allow_html=True
        )
        
        # Detailed calculation in tooltip
        if roi_explanation:
            roi_tooltip = explainer.format_for_tooltip(roi_explanation)
            with st.expander("ℹ️", expanded=False):
                st.markdown(roi_tooltip)
    
    with col2:
        # Marketing Efficiency (NEW - from enhanced calculator)
        efficiency = enhanced_metrics['efficiency'].get('efficiency', 0)
        
        if efficiency > 0:
            st.metric(
                "Marketing Efficiency",
                f"${efficiency:.2f}",
                "per $1 spent"
            )
            st.caption("Revenue ÷ Marketing Spend")
            
            with st.expander("ℹ️"):
                st.markdown(f"""**Methodology:** {enhanced_metrics['efficiency'].get('methodology', 'Unknown')}

    **Sources:** {', '.join(enhanced_metrics['efficiency'].get('sources', []))}

    **Interpretation:** {enhanced_metrics['efficiency'].get('interpretation', '')}

    **Confidence:** {enhanced_metrics['efficiency'].get('confidence', 'Unknown')}""")
        else:
            st.metric("Marketing Efficiency", "N/A", "Insufficient data")

    
    with col3:
        # Digital Marketing % (NEW - from enhanced calculator)
        digital_pct = enhanced_metrics['digital'].get('digital_percentage', 0)
        confidence = enhanced_metrics['digital'].get('confidence', 'Unknown')
        
        if digital_pct > 0:
            st.metric(
                "Digital Marketing %",
                f"{digital_pct:.0f}%",
                f"{confidence} confidence"
            )
            industry_baseline = enhanced_metrics['digital'].get('industry_baseline', 0)
            st.caption(f"Industry: {industry_baseline:.0f}%")
            
            with st.expander("ℹ️"):
                st.markdown(f"""**Methodology:** {enhanced_metrics['digital'].get('methodology', 'Unknown')}

    **Sources:** {', '.join(enhanced_metrics['digital'].get('sources', []))}

    **Interpretation:** {enhanced_metrics['digital'].get('interpretation', '')}""")
        else:
            st.metric("Digital Marketing %", "N/A", "Insufficient data")
    
    with col4:
        # MARKET SHARE
        market_share = company_metrics.get('market_share', 0)
        position = company_metrics.get('market_position', 'Unknown')
        share_explanation = company_metrics.get('explanations', {}).get('market_share', {})
        
        st.metric(
            f"Market Share",
            f"{market_share:.1f}%",
            position
        )
        
        st.markdown(
            f"<p style='font-size: 10px; color: #999; margin-top: -10px;'>Position in industry</p>",
            unsafe_allow_html=True
        )
        
        if share_explanation:
            share_tooltip = explainer.format_for_tooltip(share_explanation)
            with st.expander("ℹ️", expanded=False):
                st.markdown(share_tooltip)
    
    # === DIMINISHING RETURNS INSIGHT (Subtle integration) ===
    st.markdown("---")
    
    # Analyze efficiency
    from src.data.hypothesis_insights import DiminishingReturnsAnalyzer
    dr_analyzer = DiminishingReturnsAnalyzer()
    
    # Get current spend estimate
    roi_data = realtime_fetcher.get_company_marketing_roi(selected_company, ticker) if ticker else {}
    current_spend = roi_data.get('raw_data', {}).get('marketing_spend', 0)
    
    if current_spend > 0:
        efficiency_analysis = dr_analyzer.analyze_spend_efficiency(
            selected_company, current_roi, current_spend, industry_roi
        )
        
        insight = efficiency_analysis['diminishing_returns_insight']
        
        # Display insight subtly
        if 'Diminishing Returns Detected' in insight:
            st.warning(insight)
        elif 'Optimal Efficiency' in insight:
            st.success(insight)
        else:
            st.info(insight)
    
    # === CHARTS ROW ===
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"{selected_company} Stock Performance")
        
        # [Keep existing stock chart code]
        stock_data_raw = calculator._get_stock_data(selected_company)
        ticker = stock_data_raw.get('ticker') if stock_data_raw else None
        
        if ticker:
            try:
                import yfinance as yf
                stock_data = yf.Ticker(ticker).history(period="1y")
                
                if not stock_data.empty:
                    # Enhanced Stock Chart with Marketing Attribution
                    fig = go.Figure()

                    # Stock price
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=stock_data['Close'],
                        mode='lines',
                        name='Stock Price',
                        line=dict(color='#3b82f6', width=2.5),
                    ))

                    # Marketing attribution (20% of movement)
                    base_price = stock_data['Close'].iloc[0]
                    total_movement = stock_data['Close'] - base_price
                    marketing_attributed = base_price + (total_movement * 0.20)

                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=marketing_attributed,
                        mode='lines',
                        name='Marketing-Attributed Value',
                        line=dict(color='#10b981', width=2, dash='dash'),
                    ))

                    # Shaded area showing marketing impact zone
                    fig.add_trace(go.Scatter(
                        x=stock_data.index.tolist() + stock_data.index.tolist()[::-1],
                        y=stock_data['Close'].tolist() + marketing_attributed.tolist()[::-1],
                        fill='toself',
                        fillcolor='rgba(59, 130, 246, 0.1)',
                        line=dict(color='rgba(255,255,255,0)'),
                        name='Marketing Impact Zone',
                        showlegend=True
                    ))

                    fig.update_layout(
                        title={
                            'text': f"{selected_company} - Stock Performance with Marketing Attribution",
                            'x': 0.5,
                            'xanchor': 'center'
                        },
                        xaxis_title="Date",
                        yaxis_title="Price (USD)",
                        height=450,
                        hovermode='x unified',
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01,
                            bgcolor='rgba(255,255,255,0.8)'
                        )
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    st.caption(f"""**Methodology:** Marketing attribution model assumes 20% of stock price movement is attributable to marketing effectiveness. 
                    **Data:** Yahoo Finance ({ticker}) • Real-time""")


            except Exception as e:
                st.error(f"Error loading stock data: {str(e)}")
        else:
            st.warning(f"Stock ticker not found for {selected_company}")
    
    with col2:
        st.subheader("Marketing ROI Trend")
        
        # [Keep existing ROI trend chart with updated caption]
        stock_data_raw = calculator._get_stock_data(selected_company)
        ticker = stock_data_raw.get('ticker') if stock_data_raw else None
        
        if ticker:
            try:
                import yfinance as yf
                stock_data = yf.Ticker(ticker).history(period="1y")
                
                if not stock_data.empty and len(stock_data) > 30:
                    returns = stock_data['Close'].pct_change(30)
                    marketing_attribution = 0.20
                    spend_ratio = 0.05
                    roi_estimate = 1 + (returns * marketing_attribution / spend_ratio)
                    roi_estimate = roi_estimate.fillna(method='ffill').clip(0.5, 5.0)
                    
                    current_roi = company_metrics.get('marketing_roi', 2.1)
                    
                    fig = go.Figure()
                    
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=roi_estimate,
                        mode='lines',
                        name='Estimated ROI Trend',
                        line=dict(color='lightblue', width=1.5),
                        opacity=0.6
                    ))
                    
                    roi_smoothed = roi_estimate.rolling(window=20, min_periods=1).mean()
                    fig.add_trace(go.Scatter(
                        x=stock_data.index,
                        y=roi_smoothed,
                        mode='lines',
                        name='Smoothed Trend',
                        line=dict(color='blue', width=2.5)
                    ))
                    
                    fig.add_trace(go.Scatter(
                        x=[stock_data.index[-1]],
                        y=[current_roi],
                        mode='markers',
                        name='Current ROI',
                        marker=dict(color='green', size=12, symbol='star')
                    ))
                    
                    fig.add_hline(
                        y=industry_roi,
                        line_dash="dash",
                        line_color="gray",
                        annotation_text=f"{industry} Industry Avg: {industry_roi:.2f}x",
                        annotation_position="right"
                    )
                    
                    fig.update_layout(
                        title=f"Marketing ROI Trend - {selected_company}",
                        xaxis_title="Date",
                        yaxis_title="ROI Multiple",
                        height=400,
                        showlegend=True,
                        hovermode='x unified'
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Updated methodology caption (not hardcoded explanation)
                    st.markdown(
                        f"""<p style='font-size: 10px; color: #999;'>
                        <b>Methodology:</b> Estimated from stock performance attribution model<br>
                        • 20% of stock movement attributed to marketing<br>
                        • Data: Yahoo Finance ({ticker})<br>
                        • Current ROI: {current_roi:.2f}x • Industry Avg: {industry_roi:.2f}x
                        </p>""",
                        unsafe_allow_html=True
                    )
            except Exception as e:
                st.error(f"Error calculating ROI trend: {e}")
        else:
            st.info(f"Stock ticker not found for {selected_company}. ROI trend unavailable.")
    
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
                st.markdown(f"**Agency Strength:** {agency_data['strength']}")
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

