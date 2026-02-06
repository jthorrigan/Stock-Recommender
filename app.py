"""
Stock Recommendation Web App - Streamlit Application
Daily web crawler and AI-powered stock recommendations
Using Financial Modelling Prep (FMP) as primary and EODHD as fallback
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
    "The Economist": "https://www.economist.com/finance-and-economics/rss.xml",
}

SEC_FILINGS_API = "https://www.sec.gov/cgi-bin/browse-edgar"

# Try to load from Streamlit secrets first, fallback to environment variables
try:
    FMP_API_KEY = st.secrets.get("fmp_api_key", "")
    EODHD_API_KEY = st.secrets.get("eodhd_api_key", "")
    HF_API_KEY = st.secrets.get("hf_api_key", "")
    FRED_API_KEY = st.secrets.get("fred_api_key", "")
except:
    # Fallback to environment variables
    FMP_API_KEY = os.getenv("FMP_API_KEY", "")
    EODHD_API_KEY = os.getenv("EODHD_API_KEY", "")
    HF_API_KEY = os.getenv("HF_API_KEY", "")
    FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# API endpoints
FMP_BASE_URL = "https://financialmodelingprep.com/api/v3"
EODHD_BASE_URL = "https://eodhd.com/api"

# Expanded ticker list - Large-cap, Mid-cap, Small-cap, and ETFs
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
    """
    Retry a function with exponential backoff
    """
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
# FINANCIAL MODELLING PREP (FMP) API FUNCTIONS
# ============================================================================

@st.cache_data(ttl=3600)
def get_fmp_quote(ticker: str) -> Dict:
    """
    Fetch stock quote from Financial Modelling Prep
    Returns: price, PE ratio, market cap, etc.
    """
    if not FMP_API_KEY:
        logger.warning("FMP_API_KEY not configured")
        return {}
    
    try:
        logger.info(f"[FMP] Fetching quote for {ticker}")
        
        def fetch():
            url = f"{FMP_BASE_URL}/quote/{ticker}"
            params = {'apikey': FMP_API_KEY}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data[0] if isinstance(data, list) and len(data) > 0 else data
        
        data = retry_with_backoff(fetch, max_retries=2)
        
        if data:
            logger.info(f"✅ FMP quote for {ticker}: ${data.get('price')}")
            return data
        return {}
    
    except Exception as e:
        logger.debug(f"FMP quote failed for {ticker}: {e}")
        return {}

@st.cache_data(ttl=86400)
def get_fmp_profile(ticker: str) -> Dict:
    """
    Fetch company profile from Financial Modelling Prep
    Returns: company name, sector, industry, market cap
    """
    if not FMP_API_KEY:
        return {}
    
    try:
        logger.info(f"[FMP] Fetching profile for {ticker}")
        
        def fetch():
            url = f"{FMP_BASE_URL}/profile/{ticker}"
            params = {'apikey': FMP_API_KEY}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            return data[0] if isinstance(data, list) and len(data) > 0 else data
        
        data = retry_with_backoff(fetch, max_retries=2)
        
        if data:
            logger.info(f"✅ FMP profile for {ticker}")
            return data
        return {}
    
    except Exception as e:
        logger.debug(f"FMP profile failed for {ticker}: {e}")
        return {}

# ============================================================================
# EODHD API FUNCTIONS (FALLBACK)
# ============================================================================

@st.cache_data(ttl=3600)
def get_eodhd_quote(ticker: str) -> Dict:
    """
    Fetch real-time quote from EODHD
    """
    if not EODHD_API_KEY:
        logger.warning("EODHD_API_KEY not configured")
        return {}
    
    try:
        logger.info(f"[EODHD] Fetching quote for {ticker}")
        
        def fetch():
            url = f"{EODHD_BASE_URL}/real-time/{ticker}.US"
            params = {'api_token': EODHD_API_KEY, 'fmt': 'json'}
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        
        data = retry_with_backoff(fetch, max_retries=2)
        
        if data and data.get('close'):
            logger.info(f"✅ EODHD quote for {ticker}: ${data.get('close')}")
            return data
        return {}
    
    except Exception as e:
        logger.debug(f"EODHD quote failed for {ticker}: {e}")
        return {}

@st.cache_data(ttl=86400)
def get_eodhd_fundamentals(ticker: str) -> Dict:
    """
    Fetch fundamental data from EODHD
    Returns: market cap, dividend yield, PE ratio, etc.
    """
    if not EODHD_API_KEY:
        return {}
    
    try:
        logger.info(f"[EODHD] Fetching fundamentals for {ticker}")
        
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
        logger.debug(f"EODHD fundamentals failed for {ticker}: {e}")
        return {}

# ============================================================================
# UNIFIED METRICS FUNCTION
# ============================================================================

@st.cache_data(ttl=3600)
def get_enhanced_metrics(ticker: str) -> Dict:
    """
    Fetch stock metrics - PRIORITY ORDER:
    1. FMP (PRIMARY) - Best free tier
    2. EODHD (FALLBACK) - Reliable alternative
    """
    
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
        # PRIMARY: Financial Modelling Prep
        if FMP_API_KEY:
            try:
                logger.info(f"[1/2] Trying FMP for {ticker}")
                time.sleep(0.2)  # Small delay to avoid rate limits
                
                fmp_quote = get_fmp_quote(ticker)
                fmp_profile = get_fmp_profile(ticker)
                
                if fmp_quote:
                    # Price
                    if fmp_quote.get('price'):
                        metrics['price'] = round(float(fmp_quote.get('price')), 2)
                        logger.info(f"{ticker} price: ${metrics['price']}")
                    
                    # P/E Ratio
                    if fmp_quote.get('pe'):
                        try:
                            pe = float(fmp_quote.get('pe'))
                            if pe and pe > 0:
                                metrics['pe_ratio'] = round(pe, 2)
                        except:
                            pass
                    
                    # 52-week high/low
                    if fmp_quote.get('yearHigh'):
                        metrics['52_week_high'] = round(float(fmp_quote.get('yearHigh')), 2)
                    
                    if fmp_quote.get('yearLow'):
                        metrics['52_week_low'] = round(float(fmp_quote.get('yearLow')), 2)
                
                if fmp_profile:
                    # Market cap
                    if fmp_profile.get('mktCap'):
                        market_cap = fmp_profile.get('mktCap')
                        if market_cap >= 1e9:
                            metrics['market_cap'] = f"${market_cap/1e9:.1f}B"
                        elif market_cap >= 1e6:
                            metrics['market_cap'] = f"${market_cap/1e6:.1f}M"
                    
                    # Dividend yield
                    if fmp_profile.get('dcfValue'):
                        try:
                            metrics['eps'] = round(float(fmp_profile.get('dcfValue')), 2)
                        except:
                            pass
                
                if metrics['price'] != 'N/A':
                    metrics['data_source'] = 'FMP'
                    logger.info(f"✅ FMP complete for {ticker}")
                    return metrics
            
            except Exception as fmp_err:
                logger.debug(f"FMP failed: {fmp_err}")
        
        # FALLBACK: EODHD
        if EODHD_API_KEY:
            try:
                logger.info(f"[2/2] Trying EODHD for {ticker}")
                time.sleep(0.2)
                
                eodhd_quote = get_eodhd_quote(ticker)
                eodhd_fund = get_eodhd_fundamentals(ticker)
                
                if eodhd_quote:
                    # Price
                    if eodhd_quote.get('close'):
                        metrics['price'] = round(float(eodhd_quote.get('close')), 2)
                        logger.info(f"{ticker} price from EODHD: ${metrics['price']}")
                    
                    # 52-week high/low (from fundamental data)
                    if eodhd_fund:
                        if eodhd_fund.get('General', {}).get('52WeekHigh'):
                            metrics['52_week_high'] = round(float(eodhd_fund['General']['52WeekHigh']), 2)
                        
                        if eodhd_fund.get('General', {}).get('52WeekLow'):
                            metrics['52_week_low'] = round(float(eodhd_fund['General']['52WeekLow']), 2)
                        
                        # Market cap
                        if eodhd_fund.get('General', {}).get('MarketCapitalization'):
                            market_cap = eodhd_fund['General']['MarketCapitalization']
                            if market_cap >= 1e9:
                                metrics['market_cap'] = f"${market_cap/1e9:.1f}B"
                            elif market_cap >= 1e6:
                                metrics['market_cap'] = f"${market_cap/1e6:.1f}M"
                        
                        # Dividend yield
                        if eodhd_fund.get('Highlights', {}).get('DividendYield'):
                            try:
                                div = float(eodhd_fund['Highlights']['DividendYield'])
                                metrics['dividend_yield'] = round(div * 100, 2)
                            except:
                                pass
                        
                        # PE ratio
                        if eodhd_fund.get('Highlights', {}).get('PERatio'):
                            try:
                                metrics['pe_ratio'] = round(float(eodhd_fund['Highlights']['PERatio']), 2)
                            except:
                                pass
                
                if metrics['price'] != 'N/A':
                    metrics['data_source'] = 'EODHD'
                    logger.info(f"✅ EODHD complete for {ticker}")
                    return metrics
            
            except Exception as eodhd_err:
                logger.debug(f"EODHD failed: {eodhd_err}")
        
        logger.error(f"❌ ALL sources failed for {ticker}")
        metrics['data_source'] = 'No Data Available'
        return metrics
    
    except Exception as e:
        logger.error(f"Unexpected error for {ticker}: {str(e)}", exc_info=True)
        metrics['data_source'] = f'Error: {str(e)}'
        return metrics

@st.cache_data(ttl=86400)
def get_company_info(ticker: str) -> Dict:
    """
    Fetch company information - uses FMP first, then EODHD
    """
    try:
        logger.info(f"Fetching company info for {ticker}")
        
        # Try FMP first
        if FMP_API_KEY:
            try:
                fmp_profile = get_fmp_profile(ticker)
                if fmp_profile and fmp_profile.get('companyName'):
                    market_cap = fmp_profile.get('mktCap', 0)
                    market_cap_str = 'Unknown'
                    if market_cap and market_cap > 0:
                        if market_cap >= 1e9:
                            market_cap_str = f"${market_cap/1e9:.1f}B"
                        elif market_cap >= 1e6:
                            market_cap_str = f"${market_cap/1e6:.1f}M"
                    
                    return {
                        'ticker': ticker,
                        'name': fmp_profile.get('companyName', ticker),
                        'sector': fmp_profile.get('sector', 'Unknown'),
                        'industry': fmp_profile.get('industry', 'Unknown'),
                        'description': fmp_profile.get('description', ''),
                        'market_cap': market_cap_str,
                        'market_cap_numeric': market_cap,
                        'category': TICKER_CATEGORIES.get(ticker, 'Unknown'),
                        'data_source': 'FMP'
                    }
            except Exception as e:
                logger.debug(f"FMP company info failed: {e}")
        
        # Fallback to EODHD
        if EODHD_API_KEY:
            try:
                eodhd_fund = get_eodhd_fundamentals(ticker)
                if eodhd_fund and eodhd_fund.get('General'):
                    general = eodhd_fund['General']
                    market_cap = general.get('MarketCapitalization', 0)
                    market_cap_str = 'Unknown'
                    if market_cap and market_cap > 0:
                        if market_cap >= 1e9:
                            market_cap_str = f"${market_cap/1e9:.1f}B"
                        elif market_cap >= 1e6:
                            market_cap_str = f"${market_cap/1e6:.1f}M"
                    
                    return {
                        'ticker': ticker,
                        'name': general.get('Name', ticker),
                        'sector': general.get('Sector', 'Unknown'),
                        'industry': general.get('Industry', 'Unknown'),
                        'description': '',
                        'market_cap': market_cap_str,
                        'market_cap_numeric': market_cap,
                        'category': TICKER_CATEGORIES.get(ticker, 'Unknown'),
                        'data_source': 'EODHD'
                    }
            except Exception as e:
                logger.debug(f"EODHD company info failed: {e}")
        
        # Fallback
        return {
            'ticker': ticker,
            'name': ticker,
            'sector': 'Unknown',
            'industry': 'Unknown',
            'market_cap': 'Unknown',
            'market_cap_numeric': 0,
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
            'market_cap_numeric': 0,
            'category': TICKER_CATEGORIES.get(ticker, 'Unknown'),
            'data_source': 'Error'
        }

@st.cache_data(ttl=86400)
def get_cape_ratio_approximation() -> Dict:
    """
    Fetch CAPE ratio from FRED API (Shiller P/E)
    """
    
    cape_data = {
        'cape_ratio': 'N/A',
        'cape_date': 'N/A',
        'cape_source': 'FRED'
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
        logger.debug(f"Error fetching CAPE ratio from FRED: {str(e)}")
    
    return cape_data

# ============================================================================
# WEB CRAWLING FUNCTIONS
# ============================================================================

def crawl_news_feeds() -> List[Dict]:
    """
    Crawl RSS feeds from major financial news sources
    """
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
                    # Title
                    title_elem = item.find('title')
                    if title_elem is None:
                        title_elem = item.find('{http://www.w3.org/2005/Atom}title')
                    title = title_elem.text if title_elem is not None and title_elem.text else 'N/A'
                    
                    # Link
                    link_elem = item.find('link')
                    if link_elem is None:
                        link_elem = item.find('{http://www.w3.org/2005/Atom}link')
                    
                    if link_elem is not None:
                        if link_elem.get('href'):
                            link = link_elem.get('href')
                        else:
                            link = link_elem.text
                    else:
                        link = ''
                    
                    # Published date
                    pub_elem = item.find('pubDate')
                    if pub_elem is None:
                        pub_elem = item.find('{http://www.w3.org/2005/Atom}published')
                    published = pub_elem.text if pub_elem is not None and pub_elem.text else datetime.now().isoformat()
                    
                    # Summary
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
    """
    Crawl SEC filings for specific companies
    """
    filings = []
    
    for ticker in company_tickers:
        try:
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
    Extract stock tickers from crawled articles
    """
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
# ANALYSIS & RECOMMENDATION ENGINE
# ============================================================================

def get_stock_fundamentals(ticker: str) -> Dict:
    """
    Fetch stock fundamentals - uses FMP or EODHD
    """
    try:
        if FMP_API_KEY:
            fmp_profile = get_fmp_profile(ticker)
            if fmp_profile:
                return fmp_profile
        
        if EODHD_API_KEY:
            eodhd_fund = get_eodhd_fundamentals(ticker)
            if eodhd_fund:
                return eodhd_fund
        
        return {}
    
    except Exception as e:
        logger.debug(f"Error fetching fundamentals for {ticker}: {str(e)}")
        return {}

def calculate_confidence_score(ticker: str, articles: List[Dict], fundamentals: Dict, category: str) -> Tuple[float, str]:
    """
    Calculate confidence score and justification
    """
    
    confidence = 0.5
    factors = []
    
    # Article count factor
    article_count = len(articles)
    if article_count >= 5:
        confidence += 0.15
        factors.append(f"Strong media coverage ({article_count} articles)")
    elif article_count >= 3:
        confidence += 0.10
        factors.append(f"Moderate media coverage ({article_count} articles)")
    else:
        factors.append(f"Limited media coverage ({article_count} articles)")
    
    # Category factor
    if category == 'Large-Cap':
        confidence += 0.10
        factors.append("Established large-cap company with proven track record")
    elif category == 'Mid-Cap':
        confidence += 0.05
        factors.append("Growing mid-cap with emerging opportunities")
    elif category == 'Small-Cap':
        confidence += 0.00
        factors.append("Small-cap with higher growth potential but more volatility")
    elif category == 'ETF':
        confidence += 0.10
        factors.append("Diversified ETF reduces individual stock risk")
    
    confidence = min(confidence, 0.95)
    confidence = max(confidence, 0.50)
    
    justification = ". ".join(factors) + "."
    
    return confidence, justification

def generate_explanation_with_free_ai(
    ticker: str,
    company_name: str,
    articles: List[Dict],
    fundamentals: Dict
) -> str:
    """
    Generate investment explanation using Hugging Face
    """
    
    try:
        article_context = "\n".join([
            f"• {a['source']}: {a['title']} - {a['summary'][:100]}..."
            for a in articles[:3]
        ])
        
        prompt = f"""Write a professional 500-word investment thesis for {company_name} ({ticker}) explaining:

1. Why the 12-24 month outlook is positive
2. Key catalysts and growth drivers
3. Why now is the right time to buy
4. Market timing and valuation perspective

Recent developments:
{article_context}

Write approximately 500 words in a professional analytical tone."""

        if HF_API_KEY:
            API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
            headers = {"Authorization": f"Bearer {HF_API_KEY}"}
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_length": 1000,
                    "temperature": 0.7,
                    "top_p": 0.9
                }
            }
            
            response = requests.post(API_URL, json=payload, headers=headers, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    generated_text = result[0].get('generated_text', '')
                    if prompt in generated_text:
                        generated_text = generated_text.replace(prompt, '').strip()
                    if generated_text:
                        return generated_text[:2000]
    
    except Exception as e:
        logger.debug(f"Error generating explanation via Hugging Face for {ticker}: {str(e)}")
    
    return generate_template_explanation(ticker, company_name, articles, fundamentals)

def generate_template_explanation(
    ticker: str,
    company_name: str,
    articles: List[Dict],
    fundamentals: Dict
) -> str:
    """
    Generate template-based explanation
    """
    
    sector = fundamentals.get('sector') or fundamentals.get('Sector', 'the technology sector')
    recent_news = articles[0]['title'] if articles else "Recent market developments"
    
    explanation = f"""
## Investment Thesis: {company_name} ({ticker})

### 12-24 Month Positive Outlook

{company_name} presents a compelling investment opportunity with significant upside potential over the next 12-24 months. 
Recent market developments, including {recent_news}, demonstrate the market's growing recognition of the company's value proposition.

### Growth Catalysts and Drivers

Several key catalysts position {company_name} for outperformance:

1. **Market Expansion**: The company is well-positioned to capture market share in growing segments with competitive advantages that should drive significant revenue growth.

2. **Operational Efficiency**: Recent developments indicate improving operational metrics and margin expansion opportunities. The company's management team has demonstrated strong execution capabilities.

3. **Strategic Positioning**: {company_name} maintains a defensible competitive position with strong brand recognition and customer loyalty aligned with long-term industry trends.

### Why Now is the Right Entry Point

Current valuation levels present an attractive risk-reward opportunity for several reasons:

1. **Market Sentiment**: Recent market volatility has created a disconnect between the company's intrinsic value and current market price, presenting a compelling entry point for long-term investors.

2. **Fundamental Strength**: The company's strong balance sheet and cash generation capabilities provide downside protection while positioned for upside capture.

3. **Timing**: Market cycles suggest we are in an advantageous entry window with accelerating growth rates indicated in forward estimates.

### Conclusion

{company_name} offers an attractive risk-reward profile for 12-24 month investors. The combination of favorable catalysts, improving fundamentals, and attractive valuation creates a compelling opportunity.

**Rating: BUY | Target Horizon: 12-24 months | Risk Level: Moderate**
"""
    
    return explanation.strip()

def generate_recommendation(
    ticker: str,
    articles: List[Dict],
    fundamentals: Dict
) -> Dict:
    """
    Generate recommendation with explanation and source links
    """
    
    company_info = get_company_info(ticker)
    company_name = company_info.get('name', ticker)
    category = company_info.get('category', 'Unknown')
    
    confidence_score, confidence_justification = calculate_confidence_score(
        ticker, articles, fundamentals, category
    )
    
    explanation = generate_explanation_with_free_ai(
        ticker,
        company_name,
        articles,
        fundamentals
    )
    
    recommendation = {
        'ticker': ticker,
        'company_name': company_name,
        'sector': company_info.get('sector', 'Unknown'),
        'category': category,
        'market_cap': company_info.get('market_cap', 'Unknown'),
        'recommendation_date': datetime.now().isoformat(),
        'rating': 'BUY',
        'price_target': 'TBD',
        'explanation': explanation,
        'sources': articles[:5],
        'confidence_score': confidence_score,
        'confidence_justification': confidence_justification,
        'risk_factors': [
            'Market volatility',
            'Competitive pressures',
            'Regulatory risks',
            'Economic cycle sensitivity'
        ]
    }
    
    return recommendation

def generate_recommendations(
    crawled_articles: List[Dict],
    num_recommendations: int = 5
) -> List[Dict]:
    """
    Generate top N stock recommendations
    """
    
    stock_mentions = extract_stock_mentions(crawled_articles)
    
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
        st.markdown(f"## {rec['ticker']} - {rec['company_name']}")
        st.markdown(f"**Category:** {rec['category']} | **Sector:** {rec['sector']} | **Rating:** {rec['rating']}")
    
    with col2:
        st.metric("Confidence", f"{rec['confidence_score']:.0%}")
    
    st.info(f"📊 **Confidence Rationale:** {rec['confidence_justification']}")
    st.markdown("---")
    st.markdown("### Investment Thesis (12-24 Month Outlook)")
    st.markdown(rec['explanation'])
    
    with st.expander("⚠️ Risk Factors"):
        for risk in rec['risk_factors']:
            st.write(f"• {risk}")
    
    st.markdown("### 📚 Sources & Further Reading")
    if rec['sources']:
        for i, source in enumerate(rec['sources'], 1):
            with st.expander(f"{i}. {source['title'][:60]}..."):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Source:** {source['source']}")
                    st.markdown(f"**Published:** {source['published']}")
                with col2:
                    st.markdown(f"[🔗 Read Full Article]({source['link']})")
                st.markdown(f"**Summary:** {source['summary']}")
    else:
        st.info("No source articles available")
    
    st.markdown("---\n")

def render_crawl_status(articles: List[Dict]):
    """Render web crawl status and metrics"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Articles Crawled", len(articles))
    with col2:
        sources = len(set(a['source'] for a in articles))
        st.metric("News Sources", sources)
    with col3:
        if articles:
            latest = max(articles, key=lambda x: x['crawled_at'])
            st.metric("Last Crawl", latest['crawled_at'][:10])
    with col4:
        st.metric("Update Status", "✅ Current" if articles else "⚠️ Pending")

def render_analytics_dashboard(articles: List[Dict], recommendations: List[Dict]):
    """Render comprehensive analytics dashboard"""
    
    st.subheader("📊 Analytics Dashboard")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Tickers Mentioned", len(extract_stock_mentions(articles)))
    with col2:
        st.metric("Avg Recommendation Confidence", f"{(sum(r['confidence_score'] for r in recommendations) / len(recommendations) * 100):.0f}%" if recommendations else "N/A")
    with col3:
        st.metric("Market Cap Range", "Large to Micro")
    with col4:
        st.metric("Asset Types", "Stocks & ETFs")
    
    st.markdown("---")
    
    st.markdown("### 📈 Stock Mentions by Market Cap Category")
    
    if articles:
        mentions = extract_stock_mentions(articles)
        
        category_data = {'Large-Cap': 0, 'Mid-Cap': 0, 'Small-Cap': 0, 'ETF': 0}
        ticker_mentions = []
        
        for ticker, articles_list in mentions.items():
            category = TICKER_CATEGORIES.get(ticker, 'Unknown')
            if category in category_data:
                category_data[category] += len(articles_list)
            ticker_mentions.append({
                'Ticker': ticker,
                'Mentions': len(articles_list),
                'Category': category
            })
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Mentions by Category**")
            category_df = pd.DataFrame(
                list(category_data.items()),
                columns=['Category', 'Mentions']
            ).sort_values('Mentions', ascending=False)
            st.bar_chart(category_df.set_index('Category'), use_container_width=True, height=300)
        
        with col2:
            st.markdown("**Top 10 Mentioned Tickers**")
            top_tickers_df = pd.DataFrame(ticker_mentions).sort_values('Mentions', ascending=False).head(10)
            st.bar_chart(top_tickers_df.set_index('Ticker')['Mentions'], use_container_width=True, height=300)
        
        st.markdown("---")
        
        st.markdown("### 📋 Detailed Ticker Mentions")
        detailed_df = pd.DataFrame(ticker_mentions).sort_values('Mentions', ascending=False)
        detailed_df['Mentions'] = detailed_df['Mentions'].astype(int)
        
        st.dataframe(
            detailed_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Ticker': st.column_config.TextColumn('Stock Ticker'),
                'Mentions': st.column_config.NumberColumn('Article Mentions', format='%d'),
                'Category': st.column_config.TextColumn('Market Cap Category')
            }
        )
    else:
        st.info("Run the web crawler to see analytics data")
    
    st.markdown("---")
    
    st.markdown("### 💹 Company Financial Metrics")
    st.markdown("*Data from FMP (primary) or EODHD (fallback)*")
    
    if articles:
        mentions = extract_stock_mentions(articles)
        metrics_data = []
        progress_bar = st.progress(0)
        
        for idx, ticker in enumerate(sorted(mentions.keys())):
            metrics = get_enhanced_metrics(ticker)
            company_info = get_company_info(ticker)
            
            metrics_data.append({
                'Ticker': ticker,
                'Company': company_info.get('name', ticker)[:20],
                'Category': TICKER_CATEGORIES.get(ticker, 'N/A'),
                'Price': metrics.get('price', 'N/A'),
                'P/E Ratio': metrics.get('pe_ratio', 'N/A'),
                '52W High': metrics.get('52_week_high', 'N/A'),
                '52W Low': metrics.get('52_week_low', 'N/A'),
                'Market Cap': metrics.get('market_cap', 'N/A'),
                'Div. Yield %': metrics.get('dividend_yield', 'N/A'),
                'Data Source': metrics.get('data_source', 'N/A')
            })
            progress_bar.progress((idx + 1) / len(mentions))
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(
                metrics_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'Ticker': st.column_config.TextColumn('Symbol', width='small'),
                    'Company': st.column_config.TextColumn('Company Name'),
                    'Category': st.column_config.TextColumn('Market Cap', width='small'),
                    'Price': st.column_config.NumberColumn('Current Price', format='$%.2f'),
                    'P/E Ratio': st.column_config.NumberColumn('P/E Ratio', format='%.2f'),
                    '52W High': st.column_config.NumberColumn('52-Week High', format='$%.2f'),
                    '52W Low': st.column_config.NumberColumn('52-Week Low', format='$%.2f'),
                    'Market Cap': st.column_config.TextColumn('Market Cap', width='small'),
                    'Div. Yield %': st.column_config.NumberColumn('Dividend Yield %', format='%.2f'),
                    'Data Source': st.column_config.TextColumn('Data Source', width='small')
                }
            )
            
            cape_info = get_cape_ratio_approximation()
            if cape_info['cape_ratio'] != 'N/A':
                st.info(f"📊 **S&P 500 CAPE Ratio (Shiller P/E):** {cape_info['cape_ratio']} (as of {cape_info['cape_date']})")
            
            st.markdown("""
            **Data Source Priority:**
            
            ✅ **FMP (Financial Modelling Prep)** - PRIMARY
            - Free tier: 250 requests/day
            - Real-time quotes, fundamentals, profiles
            - Get key: https://site.financialmodelingprep.com/
            
            ✅ **EODHD** - FALLBACK
            - End-of-day data with fundamentals
            - Market cap, dividend yield, PE ratio
            - Get key: https://eodhd.com/
            
            ✅ **FRED** - MACRO DATA
            - S&P 500 CAPE ratio
            - Get key: https://fred.stlouisfed.org/
            """)

def main():
    """Main Streamlit application"""
    
    init_session()
    
    st.title("📈 Stock Recommendation Engine")
    st.markdown("AI-powered daily stock recommendations | Using FMP & EODHD APIs")
    
    with st.sidebar:
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
                with st.spinner("Analyzing articles (FMP primary, EODHD fallback)..."):
                    recommendations = generate_recommendations(st.session_state.crawled_articles)
                    st.session_state.recommendations = recommendations
                    st.success("✅ Recommendations generated")
        
        st.divider()
        
        st.markdown("### 🧪 Debug Tools")
        test_ticker = st.text_input("Test ticker (e.g., AAPL):", value="AAPL")
        if st.button("📍 Test Single Ticker", use_container_width=True):
            st.info(f"Testing {test_ticker} with FMP & EODHD...")
            try:
                test_metrics = get_enhanced_metrics(test_ticker)
                st.write("**Metrics Retrieved:**")
                st.json(test_metrics)
            except Exception as e:
                st.error(f"Error: {str(e)}")
        
        if st.button("🗑️ Clear Cache", use_container_width=True):
            clear_cache()
            st.success("Cache cleared!")
        
        st.divider()
        num_recs = st.slider("Number of Recommendations", 1, 10, 5)
        st.divider()
        
        with st.expander("ℹ️ About"):
            st.markdown("""
            Stock Recommendation Engine using:
            - **FMP** for comprehensive stock data (primary)
            - **EODHD** as reliable fallback
            - News crawling from 7 financial sources
            - FRED for market-wide CAPE ratio
            - AI-generated investment theses
            
            **Key Features:**
            ✅ Dual API strategy for reliability
            ✅ Exponential backoff retry logic
            ✅ Smart caching (1 hour for quotes)
            ✅ Error handling and graceful fallbacks
            """)
        
        with st.expander("🔑 API Configuration"):
            st.write("**Configured APIs:**")
            if FMP_API_KEY:
                st.success("✅ FMP (Financial Modelling Prep)")
            else:
                st.warning("❌ FMP not configured")
            
            if EODHD_API_KEY:
                st.success("✅ EODHD")
            else:
                st.warning("❌ EODHD not configured")
            
            if FRED_API_KEY:
                st.success("✅ FRED (CAPE Ratio)")
            else:
                st.warning("❌ FRED not configured")
    
    # Fixed: Use hasattr instead of checking truthiness directly
    has_crawl_data = hasattr(st.session_state, 'last_crawl') and st.session_state.last_crawl is not None
    
    if has_crawl_data:
        tab1, tab2, tab3 = st.tabs(["📋 Recommendations", "📰 Crawl Data", "📊 Analytics"])
        
        with tab1:
            has_recommendations = hasattr(st.session_state, 'recommendations') and st.session_state.recommendations
            if has_recommendations:
                st.subheader(f"Top {len(st.session_state.recommendations)} Recommendations")
                for i, rec in enumerate(st.session_state.recommendations[:num_recs], 1):
                    st.markdown(f"### #{i} - {rec['company_name']} ({rec['ticker']})")
                    render_recommendation_card(rec)
            else:
                st.info("Click 'Generate Recommendations' to get started")
        
        with tab2:
            st.subheader("Web Crawl Status")
            if hasattr(st.session_state, 'crawled_articles') and st.session_state.crawled_articles:
                render_crawl_status(st.session_state.crawled_articles)
                
                st.subheader("Recent Articles")
                articles_df = pd.DataFrame(st.session_state.crawled_articles)
                if not articles_df.empty:
                    st.dataframe(
                        articles_df[['source', 'title', 'published']].head(20),
                        use_container_width=True,
                        hide_index=True
                    )
            else:
                st.info("No crawl data available")
        
        with tab3:
            if hasattr(st.session_state, 'crawled_articles') and st.session_state.crawled_articles:
                render_analytics_dashboard(
                    st.session_state.crawled_articles,
                    st.session_state.recommendations if hasattr(st.session_state, 'recommendations') else []
                )
            else:
                st.info("Run the crawler first to see analytics")
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.info("👈 Start by running the web crawler")
        with col2:
            st.info("Then generate recommendations")

if __name__ == "__main__":
    main()
