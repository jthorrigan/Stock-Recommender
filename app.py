"""
Stock Recommendation Web App - Streamlit Application
FIXED API KEY LOADING FOR STREAMLIT CLOUD
"""

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import json
import os
from functools import lru_cache
import logging
import xml.etree.ElementTree as ET
import re
import numpy as np
import time

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Set page configuration
st.set_page_config(
    page_title="Stock Recommendation Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# FIXED API KEY LOADING
# ============================================================================

# IMPORTANT: This works for BOTH local (.env) and Streamlit Cloud (secrets.toml)
FMP_API_KEY = st.secrets.get("fmp_api_key") if "fmp_api_key" in st.secrets else os.getenv("FMP_API_KEY", "")
EODHD_API_KEY = st.secrets.get("eodhd_api_key") if "eodhd_api_key" in st.secrets else os.getenv("EODHD_API_KEY", "")
HF_API_KEY = st.secrets.get("hf_api_key") if "hf_api_key" in st.secrets else os.getenv("HF_API_KEY", "")
FRED_API_KEY = st.secrets.get("fred_api_key") if "fred_api_key" in st.secrets else os.getenv("FRED_API_KEY", "")

# Store API keys source for debugging
API_SOURCE = "Streamlit Secrets" if "fmp_api_key" in st.secrets else "Environment Variables"

# Log API key status
logger.info(f"API Keys Source: {API_SOURCE}")
logger.info(f"FMP_API_KEY: {'✅ Loaded' if FMP_API_KEY else '❌ Missing'}")
logger.info(f"EODHD_API_KEY: {'✅ Loaded' if EODHD_API_KEY else '❌ Missing'}")
logger.info(f"FRED_API_KEY: {'✅ Loaded' if FRED_API_KEY else '❌ Missing'}")

# API endpoints
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
EODHD_BASE_URL = "https://eodhd.com/api"

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
    "The Economist": "https://www.economist.com/finance-and-economics/rss.xml",
}

SEC_FILINGS_API = "https://www.sec.gov/cgi-bin/browse-edgar"

# Expanded ticker list
STOCK_TICKERS = {
    'Large-Cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'BAC', 'WMT'],
    'Mid-Cap': ['CRWD', 'DDOG', 'CRM', 'ZS', 'OKTA', 'SNOW', 'SSNC', 'TTD', 'NET', 'MSTR'],
    'Small-Cap': ['UPST', 'NXTC', 'BREX', 'DASH', 'COIN', 'HOOD', 'RBLX', 'PLTR', 'SOFI', 'GDS'],
    'ETF': ['SPY', 'QQQ', 'IWM', 'XLK', 'XLV', 'XLF', 'XLE', 'ARKK', 'VTSAX', 'VGIT']
}

# Flatten for searching
ALL_TICKERS = []
TICKER_CATEGORIES = {}
for category, tickers in STOCK_TICKERS.items():
    ALL_TICKERS.extend(tickers)
    for ticker in tickers:
        TICKER_CATEGORIES[ticker] = category

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

def clear_cache():
    """Clear all Streamlit caches"""
    st.cache_data.clear()
    st.cache_resource.clear()
    logger.info("Cache cleared")

# ============================================================================
# RETRY LOGIC
# ============================================================================

def retry_with_backoff(func, max_retries=3, backoff_factor=2):
    """Retry a function with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = backoff_factor ** attempt
                logger.warning(f"Attempt {attempt + 1} failed. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} attempts failed: {str(e)}")
                return None

# ============================================================================
# FMP API FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)
def get_fmp_quote(ticker: str) -> Dict:
    """Fetch stock quote from FMP"""
    
    if not FMP_API_KEY:
        logger.error("FMP_API_KEY is empty")
        return {}
    
    try:
        logger.info(f"[FMP QUOTE] Fetching {ticker}")
        
        def fetch():
            url = f"{FMP_BASE_URL}/quote/{ticker}"
            params = {'apikey': FMP_API_KEY}
            response = requests.get(url, params=params, timeout=10)
            logger.debug(f"FMP Status: {response.status_code}")
            response.raise_for_status()
            data = response.json()
            return data[0] if isinstance(data, list) and len(data) > 0 else data
        
        data = retry_with_backoff(fetch, max_retries=2)
        
        if data and 'error' not in data:
            logger.info(f"✅ FMP quote for {ticker}: ${data.get('price')}")
            return data
        return {}
    
    except Exception as e:
        logger.error(f"FMP quote failed for {ticker}: {str(e)}")
        return {}

@st.cache_data(ttl=86400)
def get_fmp_profile(ticker: str) -> Dict:
    """Fetch company profile from FMP"""
    
    if not FMP_API_KEY:
        return {}
    
    try:
        logger.info(f"[FMP PROFILE] Fetching {ticker}")
        
        def fetch():
            url = f"{FMP_BASE_URL}/profile/{ticker}"
            params = {'apikey': FMP_API_KEY}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data[0] if isinstance(data, list) and len(data) > 0 else data
        
        data = retry_with_backoff(fetch, max_retries=2)
        
        if data:
            logger.info(f"�� FMP profile for {ticker}")
            return data
        return {}
    
    except Exception as e:
        logger.error(f"FMP profile failed for {ticker}: {str(e)}")
        return {}

# ============================================================================
# EODHD API FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)
def get_eodhd_quote(ticker: str) -> Dict:
    """Fetch real-time quote from EODHD"""
    
    if not EODHD_API_KEY:
        logger.error("EODHD_API_KEY is empty")
        return {}
    
    try:
        logger.info(f"[EODHD QUOTE] Fetching {ticker}")
        
        def fetch():
            url = f"{EODHD_BASE_URL}/real-time/{ticker}.US"
            params = {'api_token': EODHD_API_KEY, 'fmt': 'json'}
            response = requests.get(url, params=params, timeout=10)
            logger.debug(f"EODHD Status: {response.status_code}")
            response.raise_for_status()
            return response.json()
        
        data = retry_with_backoff(fetch, max_retries=2)
        
        if data and data.get('close'):
            logger.info(f"✅ EODHD quote for {ticker}: ${data.get('close')}")
            return data
        return {}
    
    except Exception as e:
        logger.error(f"EODHD quote failed for {ticker}: {str(e)}")
        return {}

@st.cache_data(ttl=86400)
def get_eodhd_fundamentals(ticker: str) -> Dict:
    """Fetch fundamental data from EODHD"""
    
    if not EODHD_API_KEY:
        return {}
    
    try:
        logger.info(f"[EODHD FUNDAMENTALS] Fetching {ticker}")
        
        def fetch():
            url = f"{EODHD_BASE_URL}/fundamentals/{ticker}.US"
            params = {'api_token': EODHD_API_KEY}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        
        data = retry_with_backoff(fetch, max_retries=2)
        
        if data:
            logger.info(f"✅ EODHD fundamentals for {ticker}")
            return data
        return {}
    
    except Exception as e:
        logger.error(f"EODHD fundamentals failed for {ticker}: {str(e)}")
        return {}

# ============================================================================
# UNIFIED METRICS FUNCTION
# ============================================================================

@st.cache_data(ttl=3600)
def get_enhanced_metrics(ticker: str) -> Dict:
    """Fetch stock metrics - PRIMARY: FMP, FALLBACK: EODHD"""
    
    metrics = {
        'ticker': ticker,
        'pe_ratio': 'N/A',
        'price': 'N/A',
        '52_week_high': 'N/A',
        '52_week_low': 'N/A',
        'market_cap': 'N/A',
        'dividend_yield': 'N/A',
        'eps': 'N/A',
        'data_source': 'None'
    }
    
    try:
        # PRIMARY: FMP
        if FMP_API_KEY:
            try:
                logger.info(f"[1/2] Trying FMP for {ticker}")
                time.sleep(0.1)
                
                fmp_quote = get_fmp_quote(ticker)
                fmp_profile = get_fmp_profile(ticker)
                
                if fmp_quote and fmp_quote.get('price'):
                    metrics['price'] = round(float(fmp_quote.get('price')), 2)
                    logger.info(f"✅ Got FMP price for {ticker}: ${metrics['price']}")
                    
                    if fmp_quote.get('pe'):
                        try:
                            metrics['pe_ratio'] = round(float(fmp_quote.get('pe')), 2)
                        except:
                            pass
                    
                    if fmp_quote.get('yearHigh'):
                        metrics['52_week_high'] = round(float(fmp_quote.get('yearHigh')), 2)
                    
                    if fmp_quote.get('yearLow'):
                        metrics['52_week_low'] = round(float(fmp_quote.get('yearLow')), 2)
                
                if fmp_profile and fmp_profile.get('mktCap'):
                    market_cap = fmp_profile.get('mktCap')
                    if market_cap >= 1e9:
                        metrics['market_cap'] = f"${market_cap/1e9:.1f}B"
                    elif market_cap >= 1e6:
                        metrics['market_cap'] = f"${market_cap/1e6:.1f}M"
                
                if metrics['price'] != 'N/A':
                    metrics['data_source'] = 'FMP ✅'
                    logger.info(f"✅✅✅ FMP SUCCESS for {ticker}")
                    return metrics
            
            except Exception as fmp_err:
                logger.error(f"FMP Error: {str(fmp_err)}")
        else:
            logger.warning("FMP_API_KEY not available")
        
        # FALLBACK: EODHD
        if EODHD_API_KEY:
            try:
                logger.info(f"[2/2] Trying EODHD for {ticker}")
                time.sleep(0.1)
                
                eodhd_quote = get_eodhd_quote(ticker)
                eodhd_fund = get_eodhd_fundamentals(ticker)
                
                if eodhd_quote and eodhd_quote.get('close'):
                    metrics['price'] = round(float(eodhd_quote.get('close')), 2)
                    logger.info(f"✅ Got EODHD price for {ticker}: ${metrics['price']}")
                    
                    if eodhd_fund:
                        general = eodhd_fund.get('General', {})
                        
                        if general.get('52WeekHigh'):
                            metrics['52_week_high'] = round(float(general['52WeekHigh']), 2)
                        
                        if general.get('52WeekLow'):
                            metrics['52_week_low'] = round(float(general['52WeekLow']), 2)
                        
                        if general.get('MarketCapitalization'):
                            market_cap = general['MarketCapitalization']
                            if market_cap >= 1e9:
                                metrics['market_cap'] = f"${market_cap/1e9:.1f}B"
                            elif market_cap >= 1e6:
                                metrics['market_cap'] = f"${market_cap/1e6:.1f}M"
                        
                        highlights = eodhd_fund.get('Highlights', {})
                        if highlights.get('DividendYield'):
                            try:
                                metrics['dividend_yield'] = round(float(highlights['DividendYield']) * 100, 2)
                            except:
                                pass
                        
                        if highlights.get('PERatio'):
                            try:
                                metrics['pe_ratio'] = round(float(highlights['PERatio']), 2)
                            except:
                                pass
                
                if metrics['price'] != 'N/A':
                    metrics['data_source'] = 'EODHD ✅'
                    logger.info(f"✅✅✅ EODHD SUCCESS for {ticker}")
                    return metrics
            
            except Exception as eodhd_err:
                logger.error(f"EODHD Error: {str(eodhd_err)}")
        else:
            logger.warning("EODHD_API_KEY not available")
        
        logger.error(f"❌ NO DATA for {ticker}")
        metrics['data_source'] = 'No Data Available'
        return metrics
    
    except Exception as e:
        logger.error(f"Unexpected error for {ticker}: {str(e)}", exc_info=True)
        metrics['data_source'] = f'Error: {str(e)}'
        return metrics

@st.cache_data(ttl=86400)
def get_company_info(ticker: str) -> Dict:
    """Fetch company information"""
    try:
        if FMP_API_KEY:
            fmp_profile = get_fmp_profile(ticker)
            if fmp_profile and fmp_profile.get('companyName'):
                return {
                    'ticker': ticker,
                    'name': fmp_profile.get('companyName', ticker),
                    'sector': fmp_profile.get('sector', 'Unknown'),
                    'industry': fmp_profile.get('industry', 'Unknown'),
                    'market_cap': 'N/A',
                    'category': TICKER_CATEGORIES.get(ticker, 'Unknown'),
                    'data_source': 'FMP'
                }
        
        return {
            'ticker': ticker,
            'name': ticker,
            'sector': 'Unknown',
            'industry': 'Unknown',
            'market_cap': 'Unknown',
            'category': TICKER_CATEGORIES.get(ticker, 'Unknown'),
            'data_source': 'Cache'
        }
    
    except Exception as e:
        logger.warning(f"Error fetching company info for {ticker}: {str(e)}")
        return {
            'ticker': ticker,
            'name': ticker,
            'sector': 'Unknown',
            'industry': 'Unknown',
            'market_cap': 'Unknown',
            'category': TICKER_CATEGORIES.get(ticker, 'Unknown'),
            'data_source': 'Error'
        }

@st.cache_data(ttl=86400)
def get_cape_ratio_approximation() -> Dict:
    """Fetch CAPE ratio from FRED API"""
    
    cape_data = {'cape_ratio': 'N/A', 'cape_date': 'N/A'}
    
    if not FRED_API_KEY:
        return cape_data
    
    try:
        url = f"https://api.stlouisfed.org/fred/series/MULTPL.SHILLER_PE_RATIO/observations?api_key={FRED_API_KEY}&sort_order=desc&limit=1"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'observations' in data and len(data['observations']) > 0:
                obs = data['observations'][0]
                cape_value = obs.get('value')
                if cape_value and cape_value != '.':
                    cape_data['cape_ratio'] = round(float(cape_value), 2)
                    cape_data['cape_date'] = obs.get('date', 'N/A')
    
    except Exception as e:
        logger.debug(f"Error fetching CAPE: {str(e)}")
    
    return cape_data

# ============================================================================
# WEB CRAWLING
# ============================================================================

def crawl_news_feeds() -> List[Dict]:
    """Crawl RSS feeds"""
    articles = []
    
    for source_name, feed_url in NEWS_SOURCES.items():
        try:
            logger.info(f"Crawling {source_name}...")
            response = requests.get(feed_url, timeout=10)
            response.raise_for_status()
            
            try:
                root = ET.fromstring(response.content)
            except ET.ParseError:
                logger.warning(f"Failed to parse {source_name}")
                continue
            
            items = root.findall('.//item')
            if not items:
                items = root.findall('.//{http://www.w3.org/2005/Atom}entry')
            
            for item in items[:50]:
                try:
                    title_elem = item.find('title')
                    if title_elem is None:
                        title_elem = item.find('{http://www.w3.org/2005/Atom}title')
                    title = title_elem.text if title_elem is not None else 'N/A'
                    
                    link_elem = item.find('link')
                    if link_elem is None:
                        link_elem = item.find('{http://www.w3.org/2005/Atom}link')
                    
                    link = ''
                    if link_elem is not None:
                        link = link_elem.get('href') if link_elem.get('href') else link_elem.text
                    
                    pub_elem = item.find('pubDate')
                    if pub_elem is None:
                        pub_elem = item.find('{http://www.w3.org/2005/Atom}published')
                    published = pub_elem.text if pub_elem is not None else datetime.now().isoformat()
                    
                    summary_elem = item.find('description')
                    if summary_elem is None:
                        summary_elem = item.find('{http://www.w3.org/2005/Atom}summary')
                    summary = summary_elem.text if summary_elem is not None else ''
                    summary = re.sub('<[^<]+?>', '', summary)[:500]
                    
                    articles.append({
                        'source': source_name,
                        'title': title,
                        'link': link,
                        'published': published,
                        'summary': summary,
                        'crawled_at': datetime.now().isoformat()
                    })
                
                except Exception as e:
                    logger.debug(f"Error parsing item: {str(e)}")
        
        except Exception as e:
            logger.warning(f"Error crawling {source_name}: {str(e)}")
    
    return articles

def extract_stock_mentions(articles: List[Dict]) -> Dict[str, List]:
    """Extract stock tickers from articles"""
    stock_mentions = {}
    
    for article in articles:
        text = f"{article['title']} {article['summary']}".upper()
        for ticker in ALL_TICKERS:
            if ticker in text:
                if ticker not in stock_mentions:
                    stock_mentions[ticker] = []
                stock_mentions[ticker].append(article)
    
    return stock_mentions

def get_stock_fundamentals(ticker: str) -> Dict:
    """Fetch fundamentals"""
    try:
        if FMP_API_KEY:
            return get_fmp_profile(ticker) or {}
        return {}
    except Exception as e:
        logger.debug(f"Error fetching fundamentals: {str(e)}")
        return {}

def calculate_confidence_score(ticker: str, articles: List[Dict], fundamentals: Dict, category: str) -> Tuple[float, str]:
    """Calculate confidence"""
    confidence = 0.5
    factors = []
    
    article_count = len(articles)
    if article_count >= 5:
        confidence += 0.15
        factors.append(f"Strong coverage ({article_count} articles)")
    elif article_count >= 3:
        confidence += 0.10
        factors.append(f"Moderate coverage ({article_count} articles)")
    
    if category == 'Large-Cap':
        confidence += 0.10
    elif category == 'ETF':
        confidence += 0.10
    
    confidence = min(0.95, max(0.50, confidence))
    justification = ". ".join(factors) + "." if factors else "Stock mentioned in news"
    
    return confidence, justification

def generate_template_explanation(ticker: str, company_name: str, articles: List[Dict], fundamentals: Dict) -> str:
    """Generate explanation"""
    recent_news = articles[0]['title'] if articles else "Recent market developments"
    
    return f"""
## Investment Thesis: {company_name} ({ticker})

### 12-24 Month Positive Outlook
{company_name} presents a compelling investment opportunity. Recent news: {recent_news}

### Growth Catalysts
1. Market expansion opportunities
2. Operational efficiency improvements
3. Strategic positioning in growth markets

### Why Now
Market volatility creates attractive entry points for long-term investors.

**Rating: BUY | Horizon: 12-24 months | Risk: Moderate**
"""

def generate_recommendation(ticker: str, articles: List[Dict], fundamentals: Dict) -> Dict:
    """Generate recommendation"""
    company_info = get_company_info(ticker)
    company_name = company_info.get('name', ticker)
    category = company_info.get('category', 'Unknown')
    
    confidence_score, confidence_justification = calculate_confidence_score(ticker, articles, fundamentals, category)
    
    return {
        'ticker': ticker,
        'company_name': company_name,
        'sector': company_info.get('sector', 'Unknown'),
        'category': category,
        'market_cap': company_info.get('market_cap', 'Unknown'),
        'recommendation_date': datetime.now().isoformat(),
        'rating': 'BUY',
        'explanation': generate_template_explanation(ticker, company_name, articles, fundamentals),
        'sources': articles[:5],
        'confidence_score': confidence_score,
        'confidence_justification': confidence_justification,
        'risk_factors': ['Market volatility', 'Competitive pressures']
    }

def generate_recommendations(crawled_articles: List[Dict], num_recommendations: int = 5) -> List[Dict]:
    """Generate recommendations"""
    stock_mentions = extract_stock_mentions(crawled_articles)
    top_tickers = sorted(stock_mentions.items(), key=lambda x: len(x[1]), reverse=True)[:num_recommendations]
    
    recommendations = []
    for ticker, articles in top_tickers:
        fundamentals = get_stock_fundamentals(ticker)
        rec = generate_recommendation(ticker, articles, fundamentals)
        recommendations.append(rec)
    
    return recommendations

# ============================================================================
# STREAMLIT UI
# ============================================================================

def main():
    """Main app"""
    init_session()
    
    st.title("📈 Stock Recommendation Engine")
    st.markdown("Using FMP & EODHD APIs")
    
    with st.sidebar:
        st.header("🔧 API Status")
        
        if FMP_API_KEY:
            st.success(f"✅ FMP Configured")
        else:
            st.error(f"❌ FMP Missing")
        
        if EODHD_API_KEY:
            st.success(f"✅ EODHD Configured")
        else:
            st.error(f"❌ EODHD Missing")
        
        if FRED_API_KEY:
            st.success(f"✅ FRED Configured")
        else:
            st.warning(f"⚠️ FRED Optional")
        
        st.divider()
        st.header("⚙️ Controls")
        
        if st.button("🔄 Run Web Crawler", use_container_width=True):
            with st.spinner("Crawling..."):
                articles = crawl_news_feeds()
                st.session_state.crawled_articles = articles
                st.session_state.last_crawl = datetime.now()
                st.success(f"✅ Crawled {len(articles)} articles")
        
        if st.button("📊 Generate Recommendations", use_container_width=True):
            if not hasattr(st.session_state, 'crawled_articles') or not st.session_state.crawled_articles:
                st.warning("Run crawler first")
            else:
                with st.spinner("Generating..."):
                    recommendations = generate_recommendations(st.session_state.crawled_articles)
                    st.session_state.recommendations = recommendations
                    st.success("✅ Generated")
        
        st.divider()
        st.markdown("### 🧪 Debug")
        test_ticker = st.text_input("Test ticker:", value="AAPL")
        if st.button("📍 Test Ticker"):
            with st.spinner(f"Testing {test_ticker}..."):
                metrics = get_enhanced_metrics(test_ticker)
                st.json(metrics)
        
        if st.button("🗑️ Clear Cache"):
            clear_cache()
            st.success("Cleared!")
    
    has_crawl = hasattr(st.session_state, 'last_crawl') and st.session_state.last_crawl is not None
    
    if has_crawl:
        tab1, tab2, tab3 = st.tabs(["Recommendations", "Articles", "Analytics"])
        
        with tab1:
            if hasattr(st.session_state, 'recommendations') and st.session_state.recommendations:
                for i, rec in enumerate(st.session_state.recommendations[:5], 1):
                    st.markdown(f"### #{i} {rec['ticker']} - {rec['company_name']}")
                    st.metric("Confidence", f"{rec['confidence_score']:.0%}")
                    st.markdown(rec['explanation'])
            else:
                st.info("Generate recommendations")
        
        with tab2:
            if hasattr(st.session_state, 'crawled_articles') and st.session_state.crawled_articles:
                st.dataframe(pd.DataFrame(st.session_state.crawled_articles)[['source', 'title']], use_container_width=True)
        
        with tab3:
            if hasattr(st.session_state, 'crawled_articles') and st.session_state.crawled_articles:
                mentions = extract_stock_mentions(st.session_state.crawled_articles)
                ticker_data = []
                for ticker in sorted(mentions.keys()):
                    metrics = get_enhanced_metrics(ticker)
                    ticker_data.append({
                        'Ticker': ticker,
                        'Price': metrics.get('price'),
                        'P/E': metrics.get('pe_ratio'),
                        'Market Cap': metrics.get('market_cap'),
                        'Source': metrics.get('data_source')
                    })
                st.dataframe(pd.DataFrame(ticker_data), use_container_width=True)
    else:
        st.info("👈 Run crawler first")

if __name__ == "__main__":
    main()
