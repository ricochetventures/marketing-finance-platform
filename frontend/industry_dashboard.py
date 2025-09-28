import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def create_industry_performance_dashboard():
    """Enhanced industry comparison with time period toggles"""
    
    st.header("Industry Performance Analysis")
    
    # Industry selector
    industries = {
        'Healthcare/Pharma': {
            'companies': {
                'Novartis': {'ticker': 'NVS', 'agency': 'Omnicom'},
                'Eli Lilly': {'ticker': 'LLY', 'agency': 'Publicis'},
                'Novo Nordisk': {'ticker': 'NVO', 'agency': 'WPP'},
                'AbbVie': {'ticker': 'ABBV', 'agency': 'IPG'},
                'Pfizer': {'ticker': 'PFE', 'agency': 'Publicis'},
                'Johnson & Johnson': {'ticker': 'JNJ', 'agency': 'WPP'}
            }
        },
        'Beauty & Personal Care': {
            'companies': {
                'L\'Oréal': {'ticker': 'OR.PA', 'agency': 'Publicis'},
                'Unilever': {'ticker': 'UL', 'agency': 'WPP'},
                'Procter & Gamble': {'ticker': 'PG', 'agency': 'Publicis'},
                'Estée Lauder': {'ticker': 'EL', 'agency': 'Omnicom'},
                'Shiseido': {'ticker': '4911.T', 'agency': 'Dentsu'}
            }
        },
        'Technology': {
            'companies': {
                'Apple': {'ticker': 'AAPL', 'agency': 'Multiple'},
                'Microsoft': {'ticker': 'MSFT', 'agency': 'WPP'},
                'Google': {'ticker': 'GOOGL', 'agency': 'In-house'},
                'Meta': {'ticker': 'META', 'agency': 'WPP'},
                'Amazon': {'ticker': 'AMZN', 'agency': 'Multiple'}
            }
        },
        'Beverages': {
            'companies': {
                'Coca-Cola': {'ticker': 'KO', 'agency': 'WPP'},
                'PepsiCo': {'ticker': 'PEP', 'agency': 'Omnicom'},
                'Monster Beverage': {'ticker': 'MNST', 'agency': 'Independent'},
                'Dr Pepper': {'ticker': 'KDP', 'agency': 'Publicis'}
            }
        }
    }
    
    selected_industry = st.selectbox("Select Industry:", list(industries.keys()))
    
    # Time period selector
    col1, col2 = st.columns(2)
    with col1:
        time_period = st.select_slider(
            "Analysis Period:",
            options=["1 Year", "5 Years", "10 Years"],
            value="1 Year"
        )
    
    with col2:
        show_agencies = st.checkbox("Show Agency Relationships", value=True)
    
    # Get data for selected industry
    industry_data = industries[selected_industry]
    companies_data = []
    
    # Map time period to yfinance period
    period_map = {"1 Year": "1y", "5 Years": "5y", "10 Years": "10y"}
    period = period_map[time_period]
    
    st.subheader(f"{selected_industry} Industry Performance - {time_period}")
    
    # Fetch stock data
    with st.spinner("Loading industry performance data..."):
        for company, details in industry_data['companies'].items():
            try:
                ticker = details['ticker']
                stock = yf.Ticker(ticker)
                hist = stock.history(period=period)
                
                if not hist.empty:
                    # Calculate performance metrics
                    current_price = hist['Close'].iloc[-1]
                    start_price = hist['Close'].iloc[0]
                    total_return = ((current_price - start_price) / start_price) * 100
                    
                    # Calculate volatility
                    returns = hist['Close'].pct_change().dropna()
                    volatility = returns.std() * (252 ** 0.5) * 100  # Annualized
                    
                    companies_data.append({
                        'Company': company,
                        'Ticker': ticker,
                        'Agency': details['agency'],
                        'Current_Price': current_price,
                        'Total_Return': total_return,
                        'Volatility': volatility,
                        'Stock_History': hist['Close'],
                        'Dates': hist.index
                    })
                    
            except Exception as e:
                st.warning(f"Could not fetch data for {company}: {e}")
    
    if companies_data:
        # Create performance comparison chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Stock Performance Comparison")
            
            fig = go.Figure()
            
            # Agency color mapping
            agency_colors = {
                'WPP': '#FF6B6B',
                'Publicis': '#4ECDC4',
                'Omnicom': '#45B7D1',
                'IPG': '#96CEB4',
                'Dentsu': '#FFEAA7',
                'Multiple': '#DDA0DD',
                'Independent': '#98D8C8',
                'In-house': '#F7DC6F'
            }
            
            for company_data in companies_data:
                company = company_data['Company']
                agency = company_data['Agency']
                color = agency_colors.get(agency, '#95A5A6')
                
                # Normalize stock prices to start at 100 for comparison
                stock_history = company_data['Stock_History']
                normalized_prices = (stock_history / stock_history.iloc[0]) * 100
                
                fig.add_trace(go.Scatter(
                    x=company_data['Dates'],
                    y=normalized_prices,
                    mode='lines',
                    name=f"{company} ({agency})" if show_agencies else company,
                    line=dict(color=color, width=2),
                    hovertemplate=f"<b>{company}</b><br>" +
                                f"Agency: {agency}<br>" +
                                "Date: %{x}<br>" +
                                "Normalized Price: %{y:.1f}<br>" +
                                "<extra></extra>"
                ))
            
            fig.update_layout(
                title=f"{selected_industry} Stock Performance ({time_period})",
                xaxis_title="Date",
                yaxis_title="Normalized Price (Base = 100)",
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Agency Distribution")
            
            # Agency distribution pie chart
            agency_counts = {}
            for company_data in companies_data:
                agency = company_data['Agency']
                agency_counts[agency] = agency_counts.get(agency, 0) + 1
            
            fig_pie = px.pie(
                values=list(agency_counts.values()),
                names=list(agency_counts.keys()),
                title=f"Agency Market Share - {selected_industry}",
                color_discrete_map=agency_colors
            )
            fig_pie.update_layout(height=500)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Performance metrics table
        st.subheader("Detailed Performance Metrics")
        
        # Create DataFrame for table
        df = pd.DataFrame(companies_data)
        df = df[['Company', 'Agency', 'Total_Return', 'Volatility']].copy()
        
        # Add performance ranking
        df['Performance_Rank'] = df['Total_Return'].rank(ascending=False).astype(int)
        df['Risk_Adjusted_Return'] = df['Total_Return'] / df['Volatility']
        
        # Format for display
        display_df = df.copy()
        display_df['Total_Return'] = display_df['Total_Return'].apply(lambda x: f"{x:+.1f}%")
        display_df['Volatility'] = display_df['Volatility'].apply(lambda x: f"{x:.1f}%")
        display_df['Risk_Adjusted_Return'] = display_df['Risk_Adjusted_Return'].apply(lambda x: f"{x:.2f}")
        
        # Style the dataframe
        styled_df = display_df.style.format({
            'Performance_Rank': lambda x: f"#{x}"
        }).background_gradient(subset=['Risk_Adjusted_Return'], cmap='RdYlGn')
        
        st.dataframe(styled_df, use_container_width=True)
        
        # Agency performance analysis
        st.subheader("Agency Performance Analysis")
        
        # Calculate agency performance
        agency_performance = {}
        for company_data in companies_data:
            agency = company_data['Agency']
            if agency not in agency_performance:
                agency_performance[agency] = {
                    'returns': [],
                    'volatilities': [],
                    'companies': []
                }
            
            agency_performance[agency]['returns'].append(company_data['Total_Return'])
            agency_performance[agency]['volatilities'].append(company_data['Volatility'])
            agency_performance[agency]['companies'].append(company_data['Company'])
        
        # Create agency summary
        agency_summary = []
        for agency, data in agency_performance.items():
            avg_return = sum(data['returns']) / len(data['returns'])
            avg_volatility = sum(data['volatilities']) / len(data['volatilities'])
            
            agency_summary.append({
                'Agency': agency,
                'Avg_Return': avg_return,
                'Avg_Volatility': avg_volatility,
                'Client_Count': len(data['companies']),
                'Sample_Clients': ', '.join(data['companies'][:3])
            })
        
        agency_df = pd.DataFrame(agency_summary)
        agency_df = agency_df.sort_values('Avg_Return', ascending=False)
        
        # Format agency summary
        agency_df['Avg_Return'] = agency_df['Avg_Return'].apply(lambda x: f"{x:+.1f}%")
        agency_df['Avg_Volatility'] = agency_df['Avg_Volatility'].apply(lambda x: f"{x:.1f}%")
        
        st.dataframe(agency_df, use_container_width=True)
        
        # Key insights
        st.subheader("Key Insights")
        
        best_performer = max(companies_data, key=lambda x: x['Total_Return'])
        worst_performer = min(companies_data, key=lambda x: x['Total_Return'])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                "Top Performer",
                best_performer['Company'],
                f"{best_performer['Total_Return']:+.1f}%"
            )
            st.caption(f"Agency: {best_performer['Agency']}")
        
        with col2:
            avg_return = sum(d['Total_Return'] for d in companies_data) / len(companies_data)
            st.metric(
                "Industry Average",
                f"{avg_return:+.1f}%",
                "All companies"
            )
        
        with col3:
            st.metric(
                "Worst Performer",
                worst_performer['Company'],
                f"{worst_performer['Total_Return']:+.1f}%"
            )
            st.caption(f"Agency: {worst_performer['Agency']}")
        
        # Data sources and methodology
        with st.expander("Data Sources & Methodology"):
            st.markdown("**Data Sources:**")
            st.markdown("• Stock prices: Yahoo Finance API (real-time)")
            st.markdown("• Agency relationships: Industry databases, press releases")
            st.markdown("• Performance calculations: Total return over selected period")
            
            st.markdown("**Methodology:**")
            st.markdown("• All stock prices normalized to base 100 for comparison")
            st.markdown("• Total return = (Current Price - Start Price) / Start Price × 100")
            st.markdown("• Volatility = Annualized standard deviation of daily returns")
            st.markdown("• Risk-adjusted return = Total Return / Volatility")
            
            st.markdown("**Limitations:**")
            st.markdown("• Stock performance influenced by many factors beyond marketing")
            st.markdown("• Agency attribution is estimated, not measured directly")
            st.markdown("• External market conditions affect all companies")
    
    else:
        st.error("Unable to fetch industry data. Please try again later.")

# To integrate into main app:
# In the Compare Agencies tab, replace existing content with:
# create_industry_performance_dashboard()
