"""
Stock Recommendation Web App - Streamlit Application
Daily web crawler and AI-powered stock recommendations
Uses yfinance for best free stock data access
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
        
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        
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

def is_metrics_complete(metrics: Dict) -> bool:
    """
    Check if metrics have sufficient data
    Accept data if we have price + any other metric
    
    Args:
        metrics: Dictionary with metrics
    
    Returns:
        True if metrics have price + at least 1 other value
    """
    if metrics['price'] == 'N/A':
        return False
    
    # Count how many data points we have (excluding N/A and None)
    data_count = 0
    check_fields = ['price', '52_week_high', '52_week_low', 'market_cap', 'pe_ratio', 'dividend_yield']
    
    for field in check_fields:
        if metrics.get(field) != 'N/A' and metrics.get(field) is not None and metrics.get(field) != '':
            data_count += 1
    
    # Consider complete if we have at least price + 1 other metric
    return data_count >= 2

@st.cache_data(ttl=3600)
def get_enhanced_metrics(ticker: str) -> Dict:
    """
    Fetch stock metrics - PRIORITY ORDER:
    1. yfinance (FREE - BEST for market cap + all basic data)
    2. Finnhub (FREE - 60/min, has P/E)
    3. Alpha Vantage (LIMITED - 5/min)
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary with stock metrics
    """
    
    metrics = {
        'ticker': ticker,
        'pe_ratio': 'N/A',
        'cape_ratio': 'N/A',
        'volatility': 'N/A',
        'price': 'N/A',
        '52_week_high': 'N/A',
        '52_week_low': 'N/A',
        'market_cap': 'N/A',
        'dividend_yield': 'N/A',
        'eps': 'N/A',
        'revenue': 'N/A',
        'profit_margin': 'N/A',
        'data_source': 'None'
    }
    
    try:
        logger.info(f"Fetching metrics for {ticker}")
        
        # PRIMARY: yfinance (FREE - best for comprehensive data including market cap)
        if YFINANCE_AVAILABLE:
            try:
                logger.info(f"[1/3] Trying yfinance for {ticker}")
                
                ticker_obj = yf.Ticker(ticker)
                info = ticker_obj.info
                
                if info:
                    # Price
                    if 'currentPrice' in info:
                        metrics['price'] = round(float(info.get('currentPrice')), 2)
                        logger.info(f"{ticker} price: ${metrics['price']}")
                    elif 'regularMarketPrice' in info:
                        metrics['price'] = round(float(info.get('regularMarketPrice')), 2)
                    
                    # P/E Ratio
                    if 'trailingPE' in info:
                        try:
                            pe = info.get('trailingPE')
                            if pe and pe > 0:
                                metrics['pe_ratio'] = round(float(pe), 2)
                                logger.info(f"{ticker} P/E: {metrics['pe_ratio']}")
                        except:
                            pass
                    
                    # 52-week high/low
                    if 'fiftyTwoWeekHigh' in info:
                        try:
                            metrics['52_week_high'] = round(float(info.get('fiftyTwoWeekHigh')), 2)
                        except:
                            pass
                    
                    if 'fiftyTwoWeekLow' in info:
                        try:
                            metrics['52_week_low'] = round(float(info.get('fiftyTwoWeekLow')), 2)
                        except:
                            pass
                    
                    # Market cap - THIS IS WHERE yfinance SHINES
                    if 'marketCap' in info:
                        try:
                            market_cap = info.get('marketCap')
                            if market_cap and market_cap > 0:
                                if market_cap >= 1e9:
                                    metrics['market_cap'] = f"${market_cap/1e9:.1f}B"
                                elif market_cap >= 1e6:
                                    metrics['market_cap'] = f"${market_cap/1e6:.1f}M"
                        except:
                            pass
                    
                    # Dividend yield
                    if 'trailingAnnualDividendYield' in info:
                        try:
                            div = info.get('trailingAnnualDividendYield')
                            if div and div > 0:
                                metrics['dividend_yield'] = round(float(div) * 100, 2)
                        except:
                            pass
                    
                    # EPS
                    if 'trailingEps' in info:
                        try:
                            eps = info.get('trailingEps')
                            if eps:
                                metrics['eps'] = round(float(eps), 2)
                        except:
                            pass
                    
                    if is_metrics_complete(metrics):
                        metrics['data_source'] = 'yfinance'
                        logger.info(f"✅ yfinance complete for {ticker}")
                        return metrics
                    else:
                        logger.info(f"⚠️ yfinance has price, trying Finnhub for more data")
            
            except Exception as yf_err:
                logger.debug(f"yfinance failed: {yf_err}")
        
        # SECONDARY: Finnhub (FREE - 60 calls/min)
        if FINNHUB_API_KEY and metrics['price'] != 'N/A':
            try:
                logger.info(f"[2/3] Trying Finnhub for {ticker}")
                
                quote_url = f"{FINNHUB_BASE_URL}/quote?symbol={ticker.upper()}&token={FINNHUB_API_KEY}"
                quote_response = requests.get(quote_url, timeout=5)
                
                if quote_response.status_code == 200:
                    quote_data = quote_response.json()
                    
                    # Only fill in missing data
                    if metrics['pe_ratio'] == 'N/A' and quote_data.get('pe'):
                        try:
                            metrics['pe_ratio'] = round(float(quote_data.get('pe')), 2)
                            logger.info(f"{ticker} P/E from Finnhub: {metrics['pe_ratio']}")
                        except:
                            pass
                    
                    if metrics['52_week_high'] == 'N/A' and quote_data.get('h52'):
                        metrics['52_week_high'] = round(float(quote_data.get('h52')), 2)
                    
                    if metrics['52_week_low'] == 'N/A' and quote_data.get('l52'):
                        metrics['52_week_low'] = round(float(quote_data.get('l52')), 2)
                    
                    metrics['data_source'] = 'yfinance + Finnhub'
                    logger.info(f"Finnhub supplemented data for {ticker}")
                    return metrics
            
            except Exception as fh_err:
                logger.debug(f"Finnhub failed: {fh_err}")
        
        # If we already have price from yfinance, return it
        if metrics['price'] != 'N/A':
            metrics['data_source'] = 'yfinance'
            logger.info(f"✅ Returning yfinance data for {ticker}")
            return metrics
        
        # TERTIARY: Alpha Vantage (LIMITED - 5 calls/min)
        if ALPHA_VANTAGE_API_KEY != "demo":
            try:
                logger.info(f"[3/3] Trying Alpha Vantage for {ticker}")
                
                params = {
                    'function': 'GLOBAL_QUOTE',
                    'symbol': ticker,
                    'apikey': ALPHA_VANTAGE_API_KEY
                }
                response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=10)
                data = response.json()
                
                if 'Global Quote' in data and data['Global Quote']:
                    quote = data['Global Quote']
                    
                    if quote.get('05. price'):
                        metrics['price'] = round(float(quote.get('05. price')), 2)
                    if quote.get('52WeekHigh'):
                        metrics['52_week_high'] = round(float(quote.get('52WeekHigh')), 2)
                    if quote.get('52WeekLow'):
                        metrics['52_week_low'] = round(float(quote.get('52WeekLow')), 2)
                    
                    if is_metrics_complete(metrics):
                        metrics['data_source'] = 'Alpha Vantage'
                        logger.info(f"✅ Alpha Vantage complete for {ticker}")
                        return metrics
            
            except Exception as av_err:
                logger.debug(f"Alpha Vantage failed: {av_err}")
        
        # Return what we have
        if metrics['price'] != 'N/A':
            logger.info(f"⚠️ Partial data for {ticker}: {metrics['data_source']}")
            return metrics
        
        logger.error(f"❌ ALL sources failed for {ticker}")
        metrics['data_source'] = 'No Data Available'
        return metrics
    
    except Exception as e:
        logger.error(f"Unexpected error for {ticker}: {str(e)}")
        metrics['data_source'] = f'Error: {str(e)}'
        return metrics

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
        ticker_obj = yf.Ticker(ticker)
        info = ticker_obj.info
        
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
    st.markdown("*Data from yfinance (no API key required)*")
    
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
            **⭐ Why yfinance is BETTER than Marketstack:**
            
            ✅ **yfinance**
            - ✅ No API key required
            - ✅ Includes market cap reliably
            - ✅ Includes P/E ratio, dividend yield, EPS
            - ✅ Unlimited requests
            - ✅ Company sector & industry data
            - 💰 Cost: $0/month
            
            ❌ **Marketstack free tier**
            - ❌ Only 100 requests/month
            - ❌ MISSING market cap
            - ❌ MISSING P/E ratio
            - ❌ MISSING dividend yield
            - ❌ OHLCV data only (price, open, high, low, volume)
            - 💰 Cost: $0 free, but $99+/month for useful data
            
            **Recommendation:** Use yfinance exclusively - no other API needed!
            """)

def main():
    """Main Streamlit application"""
    
    init_session()
    
    st.title("📈 Stock Recommendation Engine")
    st.markdown("AI-powered daily stock recommendations | Powered by yfinance (no API keys needed!)")
    
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
                with st.spinner("Analyzing articles..."):
                    recommendations = generate_recommendations(st.session_state.crawled_articles)
                    st.session_state.recommendations = recommendations
                    st.success("✅ Recommendations generated")
        
        st.divider()
        
        st.markdown("### 🧪 Debug Tools")
        test_ticker = st.text_input("Test ticker (e.g., AAPL):", value="AAPL")
        if st.button("📍 Test Single Ticker", use_container_width=True):
            st.info(f"Testing {test_ticker}...")
            try:
                test_metrics = get_enhanced_metrics(test_ticker)
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
            - yfinance for all stock data (no API key!)
            - News crawling from 7 financial sources
            - AI-generated investment theses
            """)
        
        with st.expander("🔑 API Configuration"):
            st.success("✅ yfinance (primary - no key needed)")
            if FINNHUB_API_KEY:
                st.success("✅ Finnhub (optional fallback)")
            if HF_API_KEY:
                st.success("✅ Hugging Face (optional AI)")
            if FRED_API_KEY:
                st.success("✅ FRED (optional macro data)")
    
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
