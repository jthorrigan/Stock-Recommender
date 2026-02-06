"""
Stock Recommendation Web App - Streamlit Application
FINAL VERSION WITH CORRECT SECRET NAMES
"""

import streamlit as st
import pandas as pd
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import json
import os
import logging
import xml.etree.ElementTree as ET
import re
import time

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Stock Recommendation Engine", page_icon="📈", layout="wide")

# ============================================================================
# API KEY LOADING - EXACT NAMES FROM SECRETS
# ============================================================================

# These match EXACTLY what's in your secrets.toml
FMP_API_KEY = st.secrets.get("fmp_api_key", "")
EODHD_API_KEY = st.secrets.get("eodhd_api_key", "")
FRED_API_KEY = st.secrets.get("FRED_API_KEY", "")
HF_API_KEY = st.secrets.get("HF_API_KEY", "")

logger.info(f"✅ FMP loaded: {bool(FMP_API_KEY)}")
logger.info(f"✅ EODHD loaded: {bool(EODHD_API_KEY)}")
logger.info(f"✅ FRED loaded: {bool(FRED_API_KEY)}")

# API endpoints
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
EODHD_BASE_URL = "https://eodhd.com/api"

# Tickers
STOCK_TICKERS = {
    'Large-Cap': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'JPM', 'BAC', 'WMT'],
    'Mid-Cap': ['CRWD', 'DDOG', 'CRM', 'ZS', 'OKTA', 'SNOW', 'SSNC', 'TTD', 'NET', 'MSTR'],
    'Small-Cap': ['UPST', 'NXTC', 'BREX', 'DASH', 'COIN', 'HOOD', 'RBLX', 'PLTR', 'SOFI', 'GDS'],
    'ETF': ['SPY', 'QQQ', 'IWM', 'XLK', 'XLV', 'XLF', 'XLE', 'ARKK', 'VTSAX', 'VGIT']
}

ALL_TICKERS = []
TICKER_CATEGORIES = {}
for category, tickers in STOCK_TICKERS.items():
    ALL_TICKERS.extend(tickers)
    for ticker in tickers:
        TICKER_CATEGORIES[ticker] = category

NEWS_SOURCES = {
    "Reuters": "https://feeds.reuters.com/finance/markets",
    "Financial Times": "https://feeds.ft.com/markets",
    "Bloomberg": "https://feeds.bloomberg.com/markets/news.rss",
    "MarketWatch": "https://feeds.marketwatch.com/marketwatch/topstories/",
    "Seeking Alpha": "https://seekingalpha.com/feed.xml",
    "CNBC": "https://feeds.cnbc.com/cnbc/financialnews/",
    "The Economist": "https://www.economist.com/finance-and-economics/rss.xml",
}

# ============================================================================
# SESSION STATE
# ============================================================================

@st.cache_resource
def init_session():
    if 'last_crawl' not in st.session_state:
        st.session_state.last_crawl = None
    if 'recommendations' not in st.session_state:
        st.session_state.recommendations = []
    if 'crawled_articles' not in st.session_state:
        st.session_state.crawled_articles = []
    return st.session_state

init_session()

def clear_cache():
    st.cache_data.clear()
    st.cache_resource.clear()

# ============================================================================
# API FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)
def get_fmp_quote(ticker: str) -> Dict:
    if not FMP_API_KEY:
        return {}
    
    try:
        url = f"{FMP_BASE_URL}/quote/{ticker}"
        params = {'apikey': FMP_API_KEY}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data[0] if isinstance(data, list) and len(data) > 0 else data
    except Exception as e:
        logger.error(f"FMP quote error: {e}")
        return {}

@st.cache_data(ttl=86400)
def get_fmp_profile(ticker: str) -> Dict:
    if not FMP_API_KEY:
        return {}
    
    try:
        url = f"{FMP_BASE_URL}/profile/{ticker}"
        params = {'apikey': FMP_API_KEY}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data[0] if isinstance(data, list) and len(data) > 0 else data
    except Exception as e:
        logger.error(f"FMP profile error: {e}")
        return {}

@st.cache_data(ttl=3600)
def get_eodhd_quote(ticker: str) -> Dict:
    if not EODHD_API_KEY:
        return {}
    
    try:
        url = f"{EODHD_BASE_URL}/real-time/{ticker}.US"
        params = {'api_token': EODHD_API_KEY, 'fmt': 'json'}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"EODHD quote error: {e}")
        return {}

@st.cache_data(ttl=86400)
def get_eodhd_fundamentals(ticker: str) -> Dict:
    if not EODHD_API_KEY:
        return {}
    
    try:
        url = f"{EODHD_BASE_URL}/fundamentals/{ticker}.US"
        params = {'api_token': EODHD_API_KEY}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"EODHD fundamentals error: {e}")
        return {}

@st.cache_data(ttl=3600)
def get_enhanced_metrics(ticker: str) -> Dict:
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
    
    # Try FMP
    if FMP_API_KEY:
        logger.info(f"[FMP] Fetching {ticker}")
        fmp_quote = get_fmp_quote(ticker)
        fmp_profile = get_fmp_profile(ticker)
        
        if fmp_quote and fmp_quote.get('price'):
            metrics['price'] = round(float(fmp_quote.get('price')), 2)
            logger.info(f"[FMP] ✅ Price: ${metrics['price']}")
            
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
            return metrics
    
    # Try EODHD
    if EODHD_API_KEY:
        logger.info(f"[EODHD] Fetching {ticker}")
        eodhd_quote = get_eodhd_quote(ticker)
        
        if eodhd_quote and eodhd_quote.get('close'):
            metrics['price'] = round(float(eodhd_quote.get('close')), 2)
            logger.info(f"[EODHD] ✅ Price: ${metrics['price']}")
            metrics['data_source'] = 'EODHD ✅'
            return metrics
    
    logger.warning(f"[API] ❌ No data for {ticker}")
    metrics['data_source'] = 'No Data Available'
    return metrics

@st.cache_data(ttl=86400)
def get_company_info(ticker: str) -> Dict:
    try:
        if FMP_API_KEY:
            fmp_profile = get_fmp_profile(ticker)
            if fmp_profile and fmp_profile.get('companyName'):
                return {
                    'ticker': ticker,
                    'name': fmp_profile.get('companyName', ticker),
                    'sector': fmp_profile.get('sector', 'Unknown'),
                    'category': TICKER_CATEGORIES.get(ticker, 'Unknown'),
                }
        
        return {
            'ticker': ticker,
            'name': ticker,
            'sector': 'Unknown',
            'category': TICKER_CATEGORIES.get(ticker, 'Unknown'),
        }
    except Exception as e:
        logger.warning(f"Company info error: {e}")
        return {
            'ticker': ticker,
            'name': ticker,
            'sector': 'Unknown',
            'category': TICKER_CATEGORIES.get(ticker, 'Unknown'),
        }

@st.cache_data(ttl=86400)
def get_cape_ratio() -> Dict:
    if not FRED_API_KEY:
        return {'cape_ratio': 'N/A', 'cape_date': 'N/A'}
    
    try:
        url = f"https://api.stlouisfed.org/fred/series/MULTPL.SHILLER_PE_RATIO/observations?api_key={FRED_API_KEY}&sort_order=desc&limit=1"
        response = requests.get(url, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'observations' in data and len(data['observations']) > 0:
                obs = data['observations'][0]
                return {
                    'cape_ratio': round(float(obs.get('value', 'N/A')), 2),
                    'cape_date': obs.get('date', 'N/A')
                }
    except Exception as e:
        logger.debug(f"CAPE error: {e}")
    
    return {'cape_ratio': 'N/A', 'cape_date': 'N/A'}

def crawl_news_feeds() -> List[Dict]:
    articles = []
    for source_name, feed_url in NEWS_SOURCES.items():
        try:
            response = requests.get(feed_url, timeout=10)
            try:
                root = ET.fromstring(response.content)
            except:
                continue
            
            items = root.findall('.//item') or root.findall('.//{http://www.w3.org/2005/Atom}entry')
            
            for item in items[:50]:
                try:
                    title_elem = item.find('title') or item.find('{http://www.w3.org/2005/Atom}title')
                    title = title_elem.text if title_elem is not None else 'N/A'
                    
                    link_elem = item.find('link') or item.find('{http://www.w3.org/2005/Atom}link')
                    link = ''
                    if link_elem is not None:
                        link = link_elem.get('href') or link_elem.text or ''
                    
                    pub_elem = item.find('pubDate') or item.find('{http://www.w3.org/2005/Atom}published')
                    published = pub_elem.text if pub_elem is not None else datetime.now().isoformat()
                    
                    summary_elem = item.find('description') or item.find('{http://www.w3.org/2005/Atom}summary')
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
                except:
                    pass
        except:
            pass
    
    return articles

def extract_stock_mentions(articles: List[Dict]) -> Dict[str, List]:
    stock_mentions = {}
    for article in articles:
        text = f"{article['title']} {article['summary']}".upper()
        for ticker in ALL_TICKERS:
            if ticker in text:
                if ticker not in stock_mentions:
                    stock_mentions[ticker] = []
                stock_mentions[ticker].append(article)
    return stock_mentions

# ============================================================================
# UI
# ============================================================================

st.title("📈 Stock Recommendation Engine")

with st.sidebar:
    st.header("🔧 API Status")
    
    col1, col2 = st.columns(2)
    with col1:
        if FMP_API_KEY:
            st.success("✅ FMP")
        else:
            st.error("❌ FMP")
    
    with col2:
        if EODHD_API_KEY:
            st.success("✅ EODHD")
        else:
            st.error("❌ EODHD")
    
    if FRED_API_KEY:
        st.success("✅ FRED")
    else:
        st.warning("⚠️ FRED")
    
    st.divider()
    st.header("⚙️ Controls")
    
    if st.button("🔄 Run Crawler", use_container_width=True):
        with st.spinner("Crawling..."):
            articles = crawl_news_feeds()
            st.session_state.crawled_articles = articles
            st.session_state.last_crawl = datetime.now()
            st.success(f"✅ {len(articles)} articles")
    
    st.divider()
    st.markdown("### 🧪 Test API")
    test_ticker = st.text_input("Ticker:", "AAPL")
    if st.button("📍 Test", use_container_width=True):
        st.info(f"Testing {test_ticker}...")
        metrics = get_enhanced_metrics(test_ticker)
        st.json(metrics)
    
    if st.button("🗑️ Clear Cache"):
        clear_cache()
        st.success("Cleared!")

st.write("---")

has_crawl = hasattr(st.session_state, 'last_crawl') and st.session_state.last_crawl
if has_crawl:
    tab1, tab2 = st.tabs(["Articles", "Analytics"])
    
    with tab1:
        if st.session_state.crawled_articles:
            df = pd.DataFrame(st.session_state.crawled_articles)
            st.dataframe(df[['source', 'title']], use_container_width=True)
    
    with tab2:
        if st.session_state.crawled_articles:
            mentions = extract_stock_mentions(st.session_state.crawled_articles)
            ticker_data = []
            for ticker in sorted(mentions.keys())[:10]:
                metrics = get_enhanced_metrics(ticker)
                ticker_data.append({
                    'Ticker': ticker,
                    'Price': metrics['price'],
                    'Source': metrics['data_source']
                })
            if ticker_data:
                st.dataframe(pd.DataFrame(ticker_data), use_container_width=True)
else:
    st.info("👈 Run crawler to start")
