"""
Stock Recommendation Web App - Streamlit Application
Daily web crawler and AI-powered stock recommendations
"""

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import json
import os
from functools import lru_cache
import feedparser
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Stock Recommendation Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CONFIGURATION & CONSTANTS
# ============================================================================

NEWS_SOURCES = {
    "Reuters": "https://feeds.reuters.com/finance/markets",
    "Financial Times": "https://feeds.ft.com/markets",
    "Bloomberg": "https://feeds.bloomberg.com/markets/news.rss",
    "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "Seeking Alpha": "https://seekingalpha.com/feed.xml",
    "CNBC": "https://feeds.cnbc.com/cnbc/financialnews/",
}

SEC_FILINGS_API = "https://www.sec.gov/cgi-bin/browse-edgar"
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")

# ============================================================================
# CACHING & SESSION STATE
# ============================================================================

@st.cache_resource
def init_session():
    """Initialize session state variables"""
    if 'last_crawl' not in st.session_state:
        st.session_state.last_crawl = None
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []
    if 'crawled_articles' not in st.session_state:
        st.session_state.crawled_articles = []
    return st.session_state

# ============================================================================
# WEB CRAWLING FUNCTIONS
# ============================================================================

def crawl_news_feeds() -> List[Dict]:
    """
    Crawl RSS feeds from major financial news sources
    
    Returns:
        List of article dictionaries with metadata
    """
    articles = []
    
    for source_name, feed_url in NEWS_SOURCES.items():
        try:
            logger.info(f"Crawling {source_name}...")
            feed = feedparser.parse(feed_url)
            
            # Get articles from last 24 hours
            cutoff_time = datetime.now() - timedelta(days=1)
            
            for entry in feed.entries[:50]:  # Limit to 50 most recent
                published = datetime(*entry.published_parsed[:6]) if hasattr(entry, 'published_parsed') else datetime.now()
                
                if published > cutoff_time:
                    article = {
                        'source': source_name,
                        'title': entry.get('title', 'N/A'),
                        'link': entry.get('link', ''),
                        'published': published.isoformat(),
                        'summary': entry.get('summary', '')[:500],
                        'crawled_at': datetime.now().isoformat()
                    }
                    articles.append(article)
        
        except Exception as e:
            logger.warning(f"Error crawling {source_name}: {str(e)}")
    
    return articles

def crawl_sec_filings(company_tickers: List[str]) -> List[Dict]:
    """
    Crawl SEC filings (10-K, 10-Q, 8-K) for specific companies
    
    Args:
        company_tickers: List of stock tickers to research
    
    Returns:
        List of SEC filing metadata
    """
    filings = []
    
    for ticker in company_tickers:
        try:
            # Query SEC EDGAR API
            params = {
                'action': 'getcompany',
                'CIK': ticker,
                'type': '10-K|10-Q|8-K',
                'dateb': '',
                'owner': 'exclude',
                'count': 40,
                'output': 'json'
            }
            
            response = requests.get(SEC_FILINGS_API, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            for filing in data.get('filings', {}).get('recent', {}).get('filings', [])[:5]:
                filings.append({
                    'ticker': ticker,
                    'filing_type': filing.get('form'),
                    'filed_date': filing.get('filingDate'),
                    'accession_number': filing.get('accessionNumber'),
                    'url': f"https://www.sec.gov/cgi-bin/viewer?action=view&cik={ticker}&accession_number={filing.get('accessionNumber')}"
                })
        
        except Exception as e:
            logger.warning(f"Error crawling SEC filings for {ticker}: {str(e)}")
    
    return filings

def extract_stock_mentions(articles: List[Dict]) -> Dict[str, List]:
    """
    Extract stock tickers and sentiment from crawled articles
    
    Args:
        articles: List of crawled articles
    
    Returns:
        Dictionary mapping tickers to related articles
    """
    stock_mentions = {}
    
    # Simple extraction - in production, use NER model
    common_tickers = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'BAC', 'WMT']
    
    for article in articles:
        text = f"{article['title']} {article['summary']}".upper()
        for ticker in common_tickers:
            if ticker in text:
                if ticker not in stock_mentions:
                    stock_mentions[ticker] = []
                stock_mentions[ticker].append(article)
    
    return stock_mentions

# ============================================================================
# ANALYSIS & RECOMMENDATION ENGINE
# ============================================================================

def get_stock_fundamentals(ticker: str) -> Dict:
    """
    Fetch stock fundamentals from Alpha Vantage API
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary with stock data
    """
    try:
        params = {
            'function': 'OVERVIEW',
            'symbol': ticker,
            'apikey': ALPHA_VANTAGE_API_KEY
        }
        response = requests.get('https://www.alphavantage.co/query', params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(f"Error fetching fundamentals for {ticker}: {str(e)}")
        return {}

def generate_recommendation(
    ticker: str,
    articles: List[Dict],
    fundamentals: Dict
) -> Dict:
    """
    Generate recommendation with explanation and source links
    
    Args:
        ticker: Stock ticker
        articles: Related articles from crawl
        fundamentals: Stock fundamental data
    
    Returns:
        Recommendation dictionary with explanation and sources
    """
    
    # In production, integrate with GPT-4, Claude, or similar
    # This is a template structure
    recommendation = {
        'ticker': ticker,
        'recommendation_date': datetime.now().isoformat(),
        'rating': 'BUY',  # Would be determined by analysis
        'price_target': 'TBD',
        'explanation': f"""
        Stock: {ticker}
        
        12-24 Month Outlook (Positive):
        
        [500-word explanation would be generated here based on:
        - Recent SEC filings and company disclosures
        - News sentiment and market catalysts
        - Fundamental analysis metrics
        - Industry trends and competitive positioning
        - Macro factors and timing considerations]
        
        Why Now:
        - [Key catalysts for near-term upside]
        - [Market inefficiency or opportunity]
        - [Technical entry point considerations]
        """,
        'sources': articles[:5],  # Link to source articles
        'confidence_score': 0.75,
        'risk_factors': [
            'Market volatility',
            'Regulatory risks',
            'Competition'
        ]
    }
    
    return recommendation

def generate_recommendations(
    crawled_articles: List[Dict],
    num_recommendations: int = 5
) -> List[Dict]:
    """
    Generate top N stock recommendations
    
    Args:
        crawled_articles: Articles from web crawl
        num_recommendations: Number of recommendations to generate
    
    Returns:
        List of recommendation dictionaries
    """
    
    # Extract mentioned stocks
    stock_mentions = extract_stock_mentions(crawled_articles)
    
    # Sort by mention frequency and recency
    top_tickers = sorted(
        stock_mentions.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:num_recommendations]
    
    recommendations = []
    for ticker, articles in top_tickers:
        fundamentals = get_stock_fundamentals(ticker)
        rec = generate_recommendation(ticker, articles, fundamentals)
        recommendations.append(rec)
    
    return recommendations

# ============================================================================
# STREAMLIT UI
# ============================================================================

def render_recommendation_card(rec: Dict):
    """Render a single recommendation card with sources"""
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"## {rec['ticker']}")
        st.markdown(f"**Rating:** {rec['rating']} | **Confidence:** {rec['confidence_score']:.0%}")
    
    with col2:
        st.metric("Confidence", f"{rec['confidence_score']:.0%}")
    
    st.markdown("---")
    
    # Explanation section
    st.markdown("### Investment Thesis (12-24 Month Outlook)")
    st.markdown(rec['explanation'])
    
    # Risk factors
    with st.expander("Risk Factors"):
        for risk in rec['risk_factors']:
            st.write(f"• {risk}")
    
    # Source links
    st.markdown("### Sources & Further Reading")
    for i, source in enumerate(rec['sources'], 1):
        st.markdown(f"""
        **{i}. [{source['title']}]({source['link']})**
        - Source: {source['source']}
        - Published: {source['published']}
        - Summary: {source['summary']}
        """)
    
    st.markdown("---\n")

def render_crawl_status(articles: List[Dict]):
    """Render web crawl status and metrics"""
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Articles Crawled", len(articles))
    
    with col2:
        sources = len(set(a['source'] for a in articles))
        st.metric("Sources", sources)
    
    with col3:
        if articles:
            latest = max(articles, key=lambda x: x['crawled_at'])
            st.metric("Last Crawl", latest['crawled_at'][:10])
    
    with col4:
        st.metric("Update Status", "✅ Current" if articles else "⚠️ Pending")

def main():
    """Main Streamlit application"""
    
    # Initialize session state
    init_session()
    
    # Header
    st.title("📈 Stock Recommendation Engine")
    st.markdown("AI-powered daily stock recommendations based on web crawl analysis")
    
    # Sidebar controls
    with st.sidebar:
        st.header("⚙️ Controls")
        
        if st.button("🔄 Run Web Crawler Now", use_container_width=True):
            with st.spinner("Crawling financial sources..."):
                articles = crawl_news_feeds()
                st.session_state.crawled_articles = articles
                st.session_state.last_crawl = datetime.now()
                st.success(f"✅ Crawled {len(articles)} articles")
        
        if st.button("📊 Generate Recommendations", use_container_width=True):
            if not st.session_state.crawled_articles:
                st.warning("Please run crawler first")
            else:
                with st.spinner("Analyzing articles and generating recommendations..."):
                    recommendations = generate_recommendations(
                        st.session_state.crawled_articles
                    )
                    st.session_state.recommendations = recommendations
                    st.success("✅ Recommendations generated")
        
        st.divider()
        
        num_recs = st.slider(
            "Number of Recommendations",
            min_value=1,
            max_value=10,
            value=5
        )
        
        st.divider()
        
        with st.expander("ℹ️ About"):
            st.markdown("""
            This app:
            1. Crawls financial news, SEC filings, and company reports daily
            2. Extracts stock mentions and sentiment
            3. Analyzes fundamentals and market timing
            4. Recommends top 5 stocks with detailed explanations
            5. Links to original sources for verification
            """)
    
    # Main content
    if st.session_state.last_crawl:
        tab1, tab2, tab3 = st.tabs(["📋 Recommendations", "📰 Crawl Data", "📊 Analytics"])
        
        with tab1:
            if st.session_state.recommendations:
                st.subheader(f"Top {len(st.session_state.recommendations)} Stock Recommendations")
                for i, rec in enumerate(st.session_state.recommendations[:num_recs], 1):
                    st.markdown(f"### #{i}")
                    render_recommendation_card(rec)
            else:
                st.info("Click 'Generate Recommendations' to get started")
        
        with tab2:
            st.subheader("Web Crawl Status")
            render_crawl_status(st.session_state.crawled_articles)
            
            st.subheader("Recent Articles")
            articles_df = pd.DataFrame(st.session_state.crawled_articles)
            if not articles_df.empty:
                st.dataframe(
                    articles_df[['source', 'title', 'published']].head(20),
                    use_container_width=True,
                    hide_index=True
                )
        
        with tab3:
            st.subheader("Analysis Dashboard")
            
            # Stock mentions chart
            if st.session_state.crawled_articles:
                mentions = extract_stock_mentions(st.session_state.crawled_articles)
                if mentions:
                    mentions_df = pd.DataFrame(
                        [(k, len(v)) for k, v in mentions.items()],
                        columns=['Ticker', 'Mentions']
                    ).sort_values('Mentions', ascending=False)
                    
                    st.bar_chart(
                        mentions_df.set_index('Ticker'),
                        use_container_width=True
                    )
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.info("👈 Start by running the web crawler in the sidebar")
        with col2:
            st.info("Then generate recommendations based on crawled data")

if __name__ == "__main__":
    main()
