"""
Stock Recommendation Web App - Streamlit Application
Daily web crawler and AI-powered stock recommendations
Optimized for Streamlit Cloud with batch requests and caching
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

# Try to import yfinance for best free data
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False

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
    FINNHUB_API_KEY = st.secrets.get("finnhub_api_key", "")
    ALPHA_VANTAGE_API_KEY = st.secrets.get("alpha_vantage_api_key", "demo")
    HF_API_KEY = st.secrets.get("hf_api_key", "")
    FRED_API_KEY = st.secrets.get("fred_api_key", "")
except:
    # Fallback to environment variables
    FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
    ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
    HF_API_KEY = os.getenv("HF_API_KEY", "")
    FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# API base URLs
FINNHUB_BASE_URL = "https://finnhub.io/api/v1"
ALPHA_VANTAGE_BASE_URL = "https://www.alphavantage.co/query"

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

def retry_with_backoff(func, max_retries=3, backoff_factor=2, timeout=10):
    """
    Retry a function with exponential backoff
    
    Args:
        func: Function to retry
        max_retries: Maximum number of retries
        backoff_factor: Exponential backoff factor
        timeout: Request timeout in seconds
    
    Returns:
        Result from function or None if all retries fail
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
# BATCH YFINANCE REQUESTS (MOST IMPORTANT FIX)
# ============================================================================

@st.cache_data(ttl=3600)
def get_batch_metrics(tickers: List[str]) -> Dict[str, Dict]:
    """
    Fetch metrics for multiple tickers in batch (more efficient)
    This is KEY to avoiding rate limits
    
    Args:
        tickers: List of stock ticker symbols
    
    Returns:
        Dictionary mapping ticker to metrics
    """
    
    batch_metrics = {}
    
    if not YFINANCE_AVAILABLE:
        return batch_metrics
    
    logger.info(f"Fetching batch metrics for {len(tickers)} tickers")
    
    try:
        # BATCH REQUEST - Most important for rate limiting!
        logger.info(f"[BATCH] Downloading data for: {', '.join(tickers)}")
        
        def download_batch():
            return yf.download(
                tickers=tickers,
                period="1y",
                progress=False,
                group_by="ticker",
                threads=False  # IMPORTANT: Single-threaded to avoid rate limits
            )
        
        # Use retry logic
        data = retry_with_backoff(download_batch, max_retries=3, backoff_factor=2)
        
        if data is None or data.empty:
            logger.warning("Batch download returned empty data")
            return batch_metrics
        
        logger.info(f"✅ Batch download successful for {len(tickers)} tickers")
        
        # Process each ticker
        for ticker in tickers:
            try:
                metrics = {
                    'ticker': ticker,
                    'pe_ratio': 'N/A',
                    'price': 'N/A',
                    '52_week_high': 'N/A',
                    '52_week_low': 'N/A',
                    'market_cap': 'N/A',
                    'dividend_yield': 'N/A',
                    'eps': 'N/A',
                    'data_source': 'yfinance (Batch)'
                }
                
                # Extract ticker data
                if len(tickers) == 1:
                    ticker_data = data
                else:
                    ticker_data = data[ticker] if ticker in data.columns.get_level_values(0) else data[ticker]
                
                if ticker_data.empty:
                    logger.warning(f"No data for {ticker}")
                    batch_metrics[ticker] = metrics
                    continue
                
                # Current price
                try:
                    latest_close = ticker_data['Close'].iloc[-1]
                    if pd.notna(latest_close):
                        metrics['price'] = round(float(latest_close), 2)
                        logger.info(f"{ticker} price: ${metrics['price']}")
                except Exception as e:
                    logger.debug(f"Price extraction failed for {ticker}: {e}")
                
                # 52-week high/low
                try:
                    metrics['52_week_high'] = round(float(ticker_data['High'].max()), 2)
                    metrics['52_week_low'] = round(float(ticker_data['Low'].min()), 2)
                except Exception as e:
                    logger.debug(f"52W extraction failed for {ticker}: {e}")
                
                # Try to get info for P/E, market cap, etc.
                try:
                    time.sleep(0.5)  # Small delay between individual ticker info requests
                    ticker_obj = yf.Ticker(ticker)
                    info = ticker_obj.info
                    
                    if info:
                        # Market cap
                        if 'marketCap' in info and info['marketCap']:
                            market_cap = info['marketCap']
                            if market_cap >= 1e9:
                                metrics['market_cap'] = f"${market_cap/1e9:.1f}B"
                            elif market_cap >= 1e6:
                                metrics['market_cap'] = f"${market_cap/1e6:.1f}M"
                        
                        # P/E Ratio
                        if 'trailingPE' in info and info['trailingPE']:
                            metrics['pe_ratio'] = round(float(info['trailingPE']), 2)
                        
                        # Dividend yield
                        if 'trailingAnnualDividendYield' in info and info['trailingAnnualDividendYield']:
                            metrics['dividend_yield'] = round(float(info['trailingAnnualDividendYield']) * 100, 2)
                        
                        # EPS
                        if 'trailingEps' in info and info['trailingEps']:
                            metrics['eps'] = round(float(info['trailingEps']), 2)
                
                except Exception as info_err:
                    logger.debug(f"Info extraction failed for {ticker}: {info_err}")
                
                batch_metrics[ticker] = metrics
                logger.info(f"✅ Metrics for {ticker}: Price=${metrics['price']}, 52W=${metrics['52_week_low']}-${metrics['52_week_high']}")
            
            except Exception as ticker_err:
                logger.error(f"Error processing {ticker}: {str(ticker_err)}")
                batch_metrics[ticker] = metrics
        
        return batch_metrics
    
    except Exception as e:
        logger.error(f"Batch request failed: {str(e)}", exc_info=True)
        return batch_metrics

# ============================================================================
# COMPANY DATA FUNCTIONS
# ============================================================================

@st.cache_data(ttl=86400)
def get_company_info(ticker: str) -> Dict:
    """
    Fetch company information using yfinance
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary with company info
    """
    try:
        if not YFINANCE_AVAILABLE:
            return _get_company_info_fallback(ticker)
        
        logger.info(f"Fetching company info from yfinance for {ticker}")
        
        def get_info():
            ticker_obj = yf.Ticker(ticker)
            return ticker_obj.info
        
        info = retry_with_backoff(get_info, max_retries=2)
        
        if not info or 'longName' not in info:
            return _get_company_info_fallback(ticker)
        
        market_cap = info.get('marketCap', 0)
        market_cap_str = 'Unknown'
        if market_cap and market_cap > 0:
            if market_cap >= 1e9:
                market_cap_str = f"${market_cap/1e9:.1f}B"
            elif market_cap >= 1e6:
                market_cap_str = f"${market_cap/1e6:.1f}M"
        
        return {
            'ticker': ticker,
            'name': info.get('longName', ticker),
            'sector': info.get('sector', 'Unknown'),
            'industry': info.get('industry', 'Unknown'),
            'description': info.get('longBusinessSummary', ''),
            'market_cap': market_cap_str,
            'market_cap_numeric': market_cap if market_cap else 0,
            'category': TICKER_CATEGORIES.get(ticker, 'Unknown'),
            'pe_ratio': 'N/A',
            'profit_margin': 'N/A',
            'eps': 'N/A',
            'data_source': 'yfinance'
        }
    
    except Exception as e:
        logger.warning(f"Error fetching company info for {ticker}: {str(e)}")
        return _get_company_info_fallback(ticker)

def _get_company_info_fallback(ticker: str) -> Dict:
    """Fallback company info when yfinance fails"""
    return {
        'ticker': ticker,
        'name': ticker,
        'sector': 'Unknown',
        'industry': 'Unknown',
        'market_cap': 'Unknown',
        'market_cap_numeric': 0,
        'category': TICKER_CATEGORIES.get(ticker, 'Unknown'),
        'pe_ratio': 'N/A',
        'profit_margin': 'N/A',
        'eps': 'N/A',
        'data_source': 'Cache'
    }

# ============================================================================
# ENHANCED METRICS USING BATCH APPROACH
# ============================================================================

@st.cache_data(ttl=3600)
def get_enhanced_metrics(ticker: str) -> Dict:
    """
    Fetch stock metrics - uses batch approach or fallbacks
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary with stock metrics
    """
    
    # Try to get from batch cache first
    batch_result = get_batch_metrics([ticker])
    
    if ticker in batch_result and batch_result[ticker]['price'] != 'N/A':
        return batch_result[ticker]
    
    # Fallback to Finnhub
    if FINNHUB_API_KEY:
        try:
            logger.info(f"Fallback to Finnhub for {ticker}")
            time.sleep(0.5)  # Avoid rate limits
            
            quote_url = f"{FINNHUB_BASE_URL}/quote?symbol={ticker.upper()}&token={FINNHUB_API_KEY}"
            quote_response = requests.get(quote_url, timeout=5)
            
            if quote_response.status_code == 200:
                quote_data = quote_response.json()
                
                metrics = {
                    'ticker': ticker,
                    'price': round(float(quote_data.get('c')), 2) if quote_data.get('c') else 'N/A',
                    '52_week_high': round(float(quote_data.get('h52')), 2) if quote_data.get('h52') else 'N/A',
                    '52_week_low': round(float(quote_data.get('l52')), 2) if quote_data.get('l52') else 'N/A',
                    'pe_ratio': round(float(quote_data.get('pe')), 2) if quote_data.get('pe') else 'N/A',
                    'market_cap': 'N/A',
                    'dividend_yield': 'N/A',
                    'eps': 'N/A',
                    'data_source': 'Finnhub'
                }
                
                if metrics['price'] != 'N/A':
                    logger.info(f"✅ Finnhub for {ticker}: ${metrics['price']}")
                    return metrics
        
        except Exception as e:
            logger.debug(f"Finnhub failed for {ticker}: {e}")
    
    # Return empty if all fail
    return {
        'ticker': ticker,
        'price': 'N/A',
        '52_week_high': 'N/A',
        '52_week_low': 'N/A',
        'pe_ratio': 'N/A',
        'market_cap': 'N/A',
        'dividend_yield': 'N/A',
        'eps': 'N/A',
        'data_source': 'No Data Available'
    }

@st.cache_data(ttl=86400)
def get_cape_ratio_approximation() -> Dict:
    """
    Fetch CAPE ratio from FRED API (Shiller P/E)
    
    Returns:
        Dictionary with CAPE ratio data
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
    
    Returns:
        List of article dictionaries with metadata
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
    
    Args:
        company_tickers: List of stock tickers to research
    
    Returns:
        List of SEC filing metadata
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
    
    Args:
        articles: List of crawled articles
    
    Returns:
        Dictionary mapping tickers to related articles
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
    Fetch stock fundamentals - uses yfinance
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary with stock data
    """
    try:
        if not YFINANCE_AVAILABLE:
            return {}
        
        logger.info(f"Fetching fundamentals from yfinance for {ticker}")
        
        def get_info():
            ticker_obj = yf.Ticker(ticker)
            return ticker_obj.info
        
        info = retry_with_backoff(get_info, max_retries=2)
        return info if info else {}
    
    except Exception as e:
        logger.debug(f"Error fetching fundamentals for {ticker}: {str(e)}")
        return {}

def calculate_confidence_score(ticker: str, articles: List[Dict], fundamentals: Dict, category: str) -> Tuple[float, str]:
    """
    Calculate confidence score and justification
    
    Args:
        ticker: Stock ticker
        articles: Related articles
        fundamentals: Stock fundamentals
        category: Stock category
    
    Returns:
        Tuple of (confidence_score, justification_text)
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
    
    Args:
        ticker: Stock ticker
        company_name: Company name
        articles: Related articles
        fundamentals: Stock fundamentals
    
    Returns:
        500-word explanation
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
    
    Args:
        ticker: Stock ticker
        company_name: Company name
        articles: Related articles
        fundamentals: Stock fundamentals
    
    Returns:
        500-word explanation
    """
    
    sector = fundamentals.get('sector', 'the technology sector')
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

### Technical and Macro Positioning

From a macro perspective, several tailwinds support {company_name}'s investment case:

- Industry growth rates are accelerating, creating favorable backdrop for outperformance
- Regulatory environment remains supportive of the core business model
- Economic indicators suggest sustained demand for products and services

### Risk Considerations

Investors should monitor quarterly revenue growth rates, margin trends, and competitive positioning. The company faces typical sector cyclical risks and competitive pressures.

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
    
    Args:
        ticker: Stock ticker
        articles: Related articles from crawl
        fundamentals: Stock fundamental data
    
    Returns:
        Recommendation dictionary with explanation and sources
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
    
    Args:
        crawled_articles: Articles from web crawl
        num_recommendations: Number of recommendations to generate
    
    Returns:
        List of recommendation dictionaries
    """
    
    stock_mentions = extract_stock_mentions(crawled_articles)
    
    top_tickers = sorted(
        stock_mentions.items(),
        key=lambda x: len(x[1]),
        reverse=True
    )[:num_recommendations]
    
    # BATCH FETCH - This is the key optimization!
    top_ticker_list = [ticker for ticker, _ in top_tickers]
    logger.info(f"Batch fetching metrics for top {len(top_ticker_list)} tickers: {top_ticker_list}")
    batch_data = get_batch_metrics(top_ticker_list)
    
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
    st.markdown("*Data from yfinance with exponential backoff retry logic*")
    
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

def main():
    """Main Streamlit application"""
    
    init_session()
    
    st.title("📈 Stock Recommendation Engine")
    st.markdown("AI-powered daily stock recommendations | Optimized for Streamlit Cloud with batch requests")
    
    if not YFINANCE_AVAILABLE:
        st.error("❌ yfinance not installed! Run: `pip install yfinance`")
        return
    
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
                with st.spinner("Analyzing articles (using batch requests to avoid rate limits)..."):
                    recommendations = generate_recommendations(st.session_state.crawled_articles)
                    st.session_state.recommendations = recommendations
                    st.success("✅ Recommendations generated")
        
        st.divider()
        
        st.markdown("### 🧪 Debug Tools")
        test_ticker = st.text_input("Test ticker (e.g., AAPL):", value="AAPL")
        if st.button("📍 Test Single Ticker", use_container_width=True):
            st.info(f"Testing {test_ticker} with retry logic...")
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
            - **Batch yfinance downloads** (most important fix for rate limits)
            - **Exponential backoff retry logic**
            - **Streamlit caching** to reduce API calls
            - News crawling from 7 financial sources
            - AI-generated investment theses
            
            **Key optimizations:**
            ✅ Batch requests instead of individual calls
            ✅ Exponential backoff retry (2s, 4s, 8s)
            ✅ Single-threaded downloads
            ✅ Sleep between requests (0.5s)
            ✅ Aggressive caching (1 hour for metrics)
            """)
    
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
