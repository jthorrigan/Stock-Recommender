"""
Stock Recommendation Web App - Streamlit Application
WITH DIAGNOSTIC OUTPUT TO HELP DEBUG API ISSUES
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
logging.basicConfig(level=logging.DEBUG)  # Changed to DEBUG
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
    "The Economist": "https://www.economist.com/finance-and-economics/rss.xml",
}

SEC_FILINGS_API = "https://www.sec.gov/cgi-bin/browse-edgar"

# ============================================================================
# API KEY LOADING WITH DIAGNOSTICS
# ============================================================================

def load_api_keys():
    """Load API keys with detailed diagnostics"""
    keys = {}
    
    # Try Streamlit secrets first
    try:
        fmp_key = st.secrets.get("fmp_api_key", "")
        eodhd_key = st.secrets.get("eodhd_api_key", "")
        hf_key = st.secrets.get("hf_api_key", "")
        fred_key = st.secrets.get("fred_api_key", "")
        
        if fmp_key or eodhd_key:
            keys['fmp'] = fmp_key
            keys['eodhd'] = eodhd_key
            keys['hf'] = hf_key
            keys['fred'] = fred_key
            keys['source'] = 'Streamlit Secrets'
            return keys
    except Exception as e:
        logger.debug(f"Streamlit secrets not available: {e}")
    
    # Fallback to environment variables
    try:
        keys['fmp'] = os.getenv("FMP_API_KEY", "")
        keys['eodhd'] = os.getenv("EODHD_API_KEY", "")
        keys['hf'] = os.getenv("HF_API_KEY", "")
        keys['fred'] = os.getenv("FRED_API_KEY", "")
        keys['source'] = 'Environment Variables (.env)'
        return keys
    except Exception as e:
        logger.error(f"Failed to load environment variables: {e}")
        return {
            'fmp': '',
            'eodhd': '',
            'hf': '',
            'fred': '',
            'source': 'NONE - NO KEYS FOUND'
        }

# Load keys
API_KEYS = load_api_keys()
FMP_API_KEY = API_KEYS.get('fmp', '')
EODHD_API_KEY = API_KEYS.get('eodhd', '')
HF_API_KEY = API_KEYS.get('hf', '')
FRED_API_KEY = API_KEYS.get('fred', '')

# API endpoints
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
EODHD_BASE_URL = "https://eodhd.com/api"

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
# RETRY LOGIC WITH EXPONENTIAL BACKOFF
# ============================================================================

def retry_with_backoff(func, max_retries=3, backoff_factor=2):
    """Retry a function with exponential backoff"""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = backoff_factor ** attempt
                logger.warning(f"Attempt {attempt + 1} failed: {str(e)}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"All {max_retries} attempts failed: {str(e)}")
                return None

# ============================================================================
# FMP API FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)
def get_fmp_quote(ticker: str) -> Dict:
    """Fetch stock quote from FMP with detailed diagnostics"""
    
    if not FMP_API_KEY or FMP_API_KEY == "YOUR_FMP_API_KEY_HERE":
        logger.error(f"FMP_API_KEY not configured or is placeholder")
        return {'error': 'FMP_API_KEY not configured'}
    
    try:
        logger.info(f"[FMP QUOTE] Fetching for {ticker}")
        logger.debug(f"FMP_API_KEY exists: {bool(FMP_API_KEY)}, length: {len(FMP_API_KEY) if FMP_API_KEY else 0}")
        
        def fetch():
            url = f"{FMP_BASE_URL}/quote/{ticker}"
            params = {'apikey': FMP_API_KEY}
            logger.debug(f"FMP URL: {url}")
            logger.debug(f"FMP Params: {params}")
            
            response = requests.get(url, params=params, timeout=10)
            logger.debug(f"FMP Response Status: {response.status_code}")
            logger.debug(f"FMP Response Headers: {response.headers}")
            
            response.raise_for_status()
            data = response.json()
            logger.debug(f"FMP Response Data: {json.dumps(data)[:500]}")
            return data[0] if isinstance(data, list) and len(data) > 0 else data
        
        data = retry_with_backoff(fetch, max_retries=2)
        
        if data and 'error' not in data:
            logger.info(f"✅ FMP quote for {ticker}: ${data.get('price')}")
            return data
        return {}
    
    except Exception as e:
        logger.error(f"FMP quote failed for {ticker}: {str(e)}", exc_info=True)
        return {}

@st.cache_data(ttl=86400)
def get_fmp_profile(ticker: str) -> Dict:
    """Fetch company profile from FMP"""
    
    if not FMP_API_KEY or FMP_API_KEY == "YOUR_FMP_API_KEY_HERE":
        return {}
    
    try:
        logger.info(f"[FMP PROFILE] Fetching for {ticker}")
        
        def fetch():
            url = f"{FMP_BASE_URL}/profile/{ticker}"
            params = {'apikey': FMP_API_KEY}
            logger.debug(f"FMP Profile URL: {url}")
            
            response = requests.get(url, params=params, timeout=10)
            logger.debug(f"FMP Profile Response Status: {response.status_code}")
            
            response.raise_for_status()
            data = response.json()
            logger.debug(f"FMP Profile Response: {json.dumps(data)[:500]}")
            return data[0] if isinstance(data, list) and len(data) > 0 else data
        
        data = retry_with_backoff(fetch, max_retries=2)
        
        if data:
            logger.info(f"✅ FMP profile for {ticker}")
            return data
        return {}
    
    except Exception as e:
        logger.error(f"FMP profile failed for {ticker}: {str(e)}", exc_info=True)
        return {}

# ============================================================================
# EODHD API FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)
def get_eodhd_quote(ticker: str) -> Dict:
    """Fetch real-time quote from EODHD"""
    
    if not EODHD_API_KEY or EODHD_API_KEY == "YOUR_EODHD_API_KEY_HERE":
        logger.error(f"EODHD_API_KEY not configured or is placeholder")
        return {}
    
    try:
        logger.info(f"[EODHD QUOTE] Fetching for {ticker}")
        logger.debug(f"EODHD_API_KEY exists: {bool(EODHD_API_KEY)}, length: {len(EODHD_API_KEY) if EODHD_API_KEY else 0}")
        
        def fetch():
            url = f"{EODHD_BASE_URL}/real-time/{ticker}.US"
            params = {'api_token': EODHD_API_KEY, 'fmt': 'json'}
            logger.debug(f"EODHD URL: {url}")
            logger.debug(f"EODHD Params: {params}")
            
            response = requests.get(url, params=params, timeout=10)
            logger.debug(f"EODHD Response Status: {response.status_code}")
            
            response.raise_for_status()
            data = response.json()
            logger.debug(f"EODHD Response: {json.dumps(data)[:500]}")
            return data
        
        data = retry_with_backoff(fetch, max_retries=2)
        
        if data and data.get('close'):
            logger.info(f"✅ EODHD quote for {ticker}: ${data.get('close')}")
            return data
        return {}
    
    except Exception as e:
        logger.error(f"EODHD quote failed for {ticker}: {str(e)}", exc_info=True)
        return {}

@st.cache_data(ttl=86400)
def get_eodhd_fundamentals(ticker: str) -> Dict:
    """Fetch fundamental data from EODHD"""
    
    if not EODHD_API_KEY or EODHD_API_KEY == "YOUR_EODHD_API_KEY_HERE":
        return {}
    
    try:
        logger.info(f"[EODHD FUNDAMENTALS] Fetching for {ticker}")
        
        def fetch():
            url = f"{EODHD_BASE_URL}/fundamentals/{ticker}.US"
            params = {'api_token': EODHD_API_KEY}
            logger.debug(f"EODHD Fundamentals URL: {url}")
            
            response = requests.get(url, params=params, timeout=10)
            logger.debug(f"EODHD Fundamentals Response Status: {response.status_code}")
            
            response.raise_for_status()
            data = response.json()
            logger.debug(f"EODHD Fundamentals Response: {json.dumps(data)[:500]}")
            return data
        
        data = retry_with_backoff(fetch, max_retries=2)
        
        if data:
            logger.info(f"✅ EODHD fundamentals for {ticker}")
            return data
        return {}
    
    except Exception as e:
        logger.error(f"EODHD fundamentals failed for {ticker}: {str(e)}", exc_info=True)
        return {}

# ============================================================================
# UNIFIED METRICS FUNCTION
# ============================================================================

@st.cache_data(ttl=3600)
def get_enhanced_metrics(ticker: str) -> Dict:
    """Fetch stock metrics with detailed diagnostics"""
    
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
    
    # Check if we have ANY keys configured
    logger.info(f"=== STARTING METRICS FETCH FOR {ticker} ===")
    logger.info(f"API Keys configured: FMP={bool(FMP_API_KEY)}, EODHD={bool(EODHD_API_KEY)}")
    logger.info(f"API Keys source: {API_KEYS.get('source', 'UNKNOWN')}")
    
    try:
        # PRIMARY: FMP
        if FMP_API_KEY and FMP_API_KEY != "YOUR_FMP_API_KEY_HERE":
            try:
                logger.info(f"[1/2] Trying FMP for {ticker}")
                time.sleep(0.2)
                
                fmp_quote = get_fmp_quote(ticker)
                fmp_profile = get_fmp_profile(ticker)
                
                logger.debug(f"FMP Quote result: {fmp_quote}")
                logger.debug(f"FMP Profile result: {fmp_profile}")
                
                if fmp_quote and 'error' not in fmp_quote:
                    if fmp_quote.get('price'):
                        metrics['price'] = round(float(fmp_quote.get('price')), 2)
                        logger.info(f"✅ FMP Price for {ticker}: ${metrics['price']}")
                    
                    if fmp_quote.get('pe'):
                        try:
                            pe = float(fmp_quote.get('pe'))
                            if pe and pe > 0:
                                metrics['pe_ratio'] = round(pe, 2)
                        except:
                            pass
                    
                    if fmp_quote.get('yearHigh'):
                        metrics['52_week_high'] = round(float(fmp_quote.get('yearHigh')), 2)
                    
                    if fmp_quote.get('yearLow'):
                        metrics['52_week_low'] = round(float(fmp_quote.get('yearLow')), 2)
                
                if fmp_profile and 'error' not in fmp_profile:
                    if fmp_profile.get('mktCap'):
                        market_cap = fmp_profile.get('mktCap')
                        if market_cap >= 1e9:
                            metrics['market_cap'] = f"${market_cap/1e9:.1f}B"
                        elif market_cap >= 1e6:
                            metrics['market_cap'] = f"${market_cap/1e6:.1f}M"
                
                if metrics['price'] != 'N/A':
                    metrics['data_source'] = 'FMP ✅'
                    logger.info(f"✅✅✅ FMP SUCCESSFUL FOR {ticker} ✅✅✅")
                    return metrics
            
            except Exception as fmp_err:
                logger.error(f"FMP Error: {str(fmp_err)}", exc_info=True)
        else:
            logger.warning(f"FMP_API_KEY not available: '{FMP_API_KEY}'")
        
        # FALLBACK: EODHD
        if EODHD_API_KEY and EODHD_API_KEY != "YOUR_EODHD_API_KEY_HERE":
            try:
                logger.info(f"[2/2] Trying EODHD for {ticker}")
                time.sleep(0.2)
                
                eodhd_quote = get_eodhd_quote(ticker)
                eodhd_fund = get_eodhd_fundamentals(ticker)
                
                logger.debug(f"EODHD Quote result: {eodhd_quote}")
                logger.debug(f"EODHD Fundamentals result: {eodhd_fund}")
                
                if eodhd_quote and eodhd_quote.get('close'):
                    metrics['price'] = round(float(eodhd_quote.get('close')), 2)
                    logger.info(f"✅ EODHD Price for {ticker}: ${metrics['price']}")
                    
                    if eodhd_fund:
                        if eodhd_fund.get('General', {}).get('52WeekHigh'):
                            metrics['52_week_high'] = round(float(eodhd_fund['General']['52WeekHigh']), 2)
                        
                        if eodhd_fund.get('General', {}).get('MarketCapitalization'):
                            market_cap = eodhd_fund['General']['MarketCapitalization']
                            if market_cap >= 1e9:
                                metrics['market_cap'] = f"${market_cap/1e9:.1f}B"
                            elif market_cap >= 1e6:
                                metrics['market_cap'] = f"${market_cap/1e6:.1f}M"
                
                if metrics['price'] != 'N/A':
                    metrics['data_source'] = 'EODHD ✅'
                    logger.info(f"✅✅✅ EODHD SUCCESSFUL FOR {ticker} ✅✅✅")
                    return metrics
            
            except Exception as eodhd_err:
                logger.error(f"EODHD Error: {str(eodhd_err)}", exc_info=True)
        else:
            logger.warning(f"EODHD_API_KEY not available: '{EODHD_API_KEY}'")
        
        logger.error(f"❌❌❌ NO DATA SOURCES WORKED FOR {ticker} ❌❌❌")
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
        if FMP_API_KEY and FMP_API_KEY != "YOUR_FMP_API_KEY_HERE":
            try:
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
            except Exception as e:
                logger.debug(f"FMP company info failed: {e}")
        
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
    
    cape_data = {
        'cape_ratio': 'N/A',
        'cape_date': 'N/A',
    }
    
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
        logger.debug(f"Error fetching CAPE ratio: {str(e)}")
    
    return cape_data

# ============================================================================
# WEB CRAWLING (unchanged)
# ============================================================================

def crawl_news_feeds() -> List[Dict]:
    """Crawl RSS feeds from major financial news sources"""
    articles = []
    
    for source_name, feed_url in NEWS_SOURCES.items():
        try:
            logger.info(f"Crawling {source_name}...")
            response = requests.get(feed_url, timeout=10)
            response.raise_for_status()
            
            try:
                root = ET.fromstring(response.content)
            except ET.ParseError:
                logger.warning(f"Failed to parse XML from {source_name}")
                continue
            
            items = root.findall('.//item')
            if not items:
                items = root.findall('.//{http://www.w3.org/2005/Atom}entry')
            
            for item in items[:50]:
                try:
                    title_elem = item.find('title')
                    if title_elem is None:
                        title_elem = item.find('{http://www.w3.org/2005/Atom}title')
                    title = title_elem.text if title_elem is not None and title_elem.text else 'N/A'
                    
                    link_elem = item.find('link')
                    if link_elem is None:
                        link_elem = item.find('{http://www.w3.org/2005/Atom}link')
                    
                    if link_elem is not None:
                        link = link_elem.get('href') if link_elem.get('href') else link_elem.text
                    else:
                        link = ''
                    
                    pub_elem = item.find('pubDate')
                    if pub_elem is None:
                        pub_elem = item.find('{http://www.w3.org/2005/Atom}published')
                    published = pub_elem.text if pub_elem is not None and pub_elem.text else datetime.now().isoformat()
                    
                    summary_elem = item.find('description')
                    if summary_elem is None:
                        summary_elem = item.find('{http://www.w3.org/2005/Atom}summary')
                    summary = summary_elem.text if summary_elem is not None and summary_elem.text else ''
                    summary = re.sub('<[^<]+?>', '', summary)[:500]
                    
                    article = {
                        'source': source_name,
                        'title': title,
                        'link': link,
                        'published': published,
                        'summary': summary,
                        'crawled_at': datetime.now().isoformat()
                    }
                    articles.append(article)
                
                except Exception as e:
                    logger.debug(f"Error parsing item from {source_name}: {str(e)}")
                    continue
        
        except Exception as e:
            logger.warning(f"Error crawling {source_name}: {str(e)}")
    
    return articles

def crawl_sec_filings(company_tickers: List[str]) -> List[Dict]:
    """Crawl SEC filings"""
    filings = []
    
    for ticker in company_tickers:
        try:
            params = {
                'action': 'getcompany',
                'CIK': ticker,
                'type': '10-K|10-Q|8-K',
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
                })
        
        except Exception as e:
            logger.warning(f"Error crawling SEC filings for {ticker}: {str(e)}")
    
    return filings

def extract_stock_mentions(articles: List[Dict]) -> Dict[str, List]:
    """Extract stock tickers from crawled articles"""
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
    """Fetch stock fundamentals"""
    try:
        if FMP_API_KEY and FMP_API_KEY != "YOUR_FMP_API_KEY_HERE":
            fmp_profile = get_fmp_profile(ticker)
            if fmp_profile:
                return fmp_profile
        return {}
    except Exception as e:
        logger.debug(f"Error fetching fundamentals for {ticker}: {str(e)}")
        return {}

def calculate_confidence_score(ticker: str, articles: List[Dict], fundamentals: Dict, category: str) -> Tuple[float, str]:
    """Calculate confidence score"""
    confidence = 0.5
    factors = []
    
    article_count = len(articles)
    if article_count >= 5:
        confidence += 0.15
        factors.append(f"Strong media coverage ({article_count} articles)")
    elif article_count >= 3:
        confidence += 0.10
        factors.append(f"Moderate media coverage ({article_count} articles)")
    else:
        factors.append(f"Limited media coverage ({article_count} articles)")
    
    if category == 'Large-Cap':
        confidence += 0.10
        factors.append("Large-cap company")
    elif category == 'Mid-Cap':
        confidence += 0.05
        factors.append("Mid-cap company")
    
    confidence = min(confidence, 0.95)
    confidence = max(confidence, 0.50)
    
    justification = ". ".join(factors) + "."
    return confidence, justification

def generate_template_explanation(ticker: str, company_name: str, articles: List[Dict], fundamentals: Dict) -> str:
    """Generate explanation"""
    return f"""
## Investment Thesis: {company_name} ({ticker})

### 12-24 Month Positive Outlook

{company_name} presents a compelling investment opportunity with significant upside potential over the next 12-24 months.

### Growth Catalysts and Drivers

Several key catalysts position {company_name} for outperformance

### Why Now is the Right Entry Point

Current valuation levels present an attractive risk-reward opportunity.

**Rating: BUY | Target Horizon: 12-24 months | Risk Level: Moderate**
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
    """Generate top N recommendations"""
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

def render_recommendation_card(rec: Dict):
    """Render recommendation card"""
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown(f"## {rec['ticker']} - {rec['company_name']}")
        st.markdown(f"**Category:** {rec['category']} | **Sector:** {rec['sector']}")
    
    with col2:
        st.metric("Confidence", f"{rec['confidence_score']:.0%}")
    
    st.info(f"📊 {rec['confidence_justification']}")
    st.markdown(rec['explanation'])

def render_crawl_status(articles: List[Dict]):
    """Render crawl status"""
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Articles", len(articles))
    with col2:
        st.metric("Sources", len(set(a['source'] for a in articles)))

def render_analytics_dashboard(articles: List[Dict], recommendations: List[Dict]):
    """Render analytics"""
    st.subheader("📊 Analytics")
    
    if articles:
        mentions = extract_stock_mentions(articles)
        
        ticker_data = []
        for idx, ticker in enumerate(sorted(mentions.keys())):
            metrics = get_enhanced_metrics(ticker)
            company_info = get_company_info(ticker)
            
            ticker_data.append({
                'Ticker': ticker,
                'Company': company_info.get('name', ticker)[:20],
                'Price': metrics.get('price', 'N/A'),
                'P/E': metrics.get('pe_ratio', 'N/A'),
                'Market Cap': metrics.get('market_cap', 'N/A'),
                'Source': metrics.get('data_source', 'N/A')
            })
        
        if ticker_data:
            st.dataframe(pd.DataFrame(ticker_data), use_container_width=True)

def main():
    """Main app"""
    init_session()
    
    st.title("📈 Stock Recommendation Engine")
    st.markdown("Using FMP & EODHD APIs")
    
    # Display API Configuration Status
    with st.sidebar:
        st.header("🔧 API Configuration")
        st.subheader("Configuration Source")
        st.info(f"Loading from: **{API_KEYS.get('source', 'UNKNOWN')}**")
        
        st.subheader("API Keys Status")
        
        if FMP_API_KEY and FMP_API_KEY != "YOUR_FMP_API_KEY_HERE":
            st.success(f"✅ FMP Configured (key length: {len(FMP_API_KEY)})")
        else:
            st.error(f"❌ FMP Not Configured: '{FMP_API_KEY[:50]}...'")
        
        if EODHD_API_KEY and EODHD_API_KEY != "YOUR_EODHD_API_KEY_HERE":
            st.success(f"✅ EODHD Configured (key length: {len(EODHD_API_KEY)})")
        else:
            st.error(f"❌ EODHD Not Configured: '{EODHD_API_KEY[:50]}...'")
        
        if FRED_API_KEY:
            st.success(f"✅ FRED Configured")
        else:
            st.warning(f"⚠️ FRED Not Configured")
        
        st.divider()
        st.header("⚙️ Controls")
        
        if st.button("🔄 Run Web Crawler Now", use_container_width=True):
            with st.spinner("Crawling financial sources..."):
                articles = crawl_news_feeds()
                st.session_state.crawled_articles = articles
                st.session_state.last_crawl = datetime.now()
                st.success(f"✅ Crawled {len(articles)} articles")
        
        if st.button("📊 Generate Recommendations", use_container_width=True):
            if not hasattr(st.session_state, 'crawled_articles') or not st.session_state.crawled_articles:
                st.warning("Please run crawler first")
            else:
                with st.spinner("Generating recommendations..."):
                    recommendations = generate_recommendations(st.session_state.crawled_articles)
                    st.session_state.recommendations = recommendations
                    st.success("✅ Recommendations generated")
        
        st.divider()
        st.markdown("### 🧪 Debug Tools")
        test_ticker = st.text_input("Test ticker:", value="AAPL")
        if st.button("📍 Test Single Ticker", use_container_width=True):
            st.info(f"Testing {test_ticker} with FMP & EODHD...")
            with st.spinner("Fetching data..."):
                test_metrics = get_enhanced_metrics(test_ticker)
                st.write("**Metrics Retrieved:**")
                st.json(test_metrics)
        
        if st.button("🗑️ Clear Cache", use_container_width=True):
            clear_cache()
            st.success("Cache cleared!")
    
    # Main content
    has_crawl_data = hasattr(st.session_state, 'last_crawl') and st.session_state.last_crawl is not None
    
    if has_crawl_data:
        tab1, tab2, tab3 = st.tabs(["📋 Recommendations", "📰 Crawl Data", "📊 Analytics"])
        
        with tab1:
            has_recommendations = hasattr(st.session_state, 'recommendations') and st.session_state.recommendations
            if has_recommendations:
                st.subheader("Recommendations")
                for i, rec in enumerate(st.session_state.recommendations[:5], 1):
                    st.markdown(f"### #{i} - {rec['company_name']} ({rec['ticker']})")
                    render_recommendation_card(rec)
            else:
                st.info("Click 'Generate Recommendations'")
        
        with tab2:
            if hasattr(st.session_state, 'crawled_articles') and st.session_state.crawled_articles:
                render_crawl_status(st.session_state.crawled_articles)
                articles_df = pd.DataFrame(st.session_state.crawled_articles)
                st.dataframe(articles_df[['source', 'title']].head(20), use_container_width=True)
        
        with tab3:
            if hasattr(st.session_state, 'crawled_articles') and st.session_state.crawled_articles:
                render_analytics_dashboard(
                    st.session_state.crawled_articles,
                    st.session_state.recommendations if hasattr(st.session_state, 'recommendations') else []
                )
    else:
        st.info("👈 Start by running the web crawler")

if __name__ == "__main__":
    main()
