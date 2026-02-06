"""
Stock Recommendation Web App - Streamlit Application
Daily web crawler and AI-powered stock recommendations
Uses multiple free data sources to minimize API requests
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
MARKETSTACK_API_KEY = os.getenv("MARKETSTACK_API_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_VANTAGE_API_KEY", "demo")
HF_API_KEY = os.getenv("HF_API_KEY", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

# API base URLs
MARKETSTACK_BASE_URL = "http://api.marketstack.com/v1"
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
    Fetch company information - uses Alpha Vantage first (limited), then Yahoo
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary with company info
    """
    try:
        # Try Alpha Vantage first (only 5 calls/min, so use sparingly)
        if ALPHA_VANTAGE_API_KEY != "demo":
            try:
                logger.info(f"Fetching company info from Alpha Vantage for {ticker}")
                params = {
                    'function': 'OVERVIEW',
                    'symbol': ticker,
                    'apikey': ALPHA_VANTAGE_API_KEY
                }
                response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if 'Name' in data and data.get('Name'):
                    return {
                        'ticker': ticker,
                        'name': data.get('Name', ticker),
                        'sector': data.get('Sector', 'Unknown'),
                        'industry': data.get('Industry', 'Unknown'),
                        'description': data.get('Description', ''),
                        'market_cap': data.get('MarketCapitalization', 'Unknown'),
                        'market_cap_numeric': 0,
                        'category': TICKER_CATEGORIES.get(ticker, 'Unknown'),
                        'pe_ratio': 'N/A',
                        'profit_margin': 'N/A',
                        'eps': 'N/A',
                        'data_source': 'Alpha Vantage'
                    }
            except Exception as e:
                logger.debug(f"Alpha Vantage company info failed: {e}")
        
        # Fallback: Use cached ticker info
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
            'pe_ratio': 'N/A',
            'profit_margin': 'N/A',
            'eps': 'N/A',
            'data_source': 'Error'
        }

@st.cache_data(ttl=3600)
def get_enhanced_metrics(ticker: str) -> Dict:
    """
    Fetch stock metrics - PRIORITY ORDER:
    1. Yahoo Finance (FREE - NO LIMITS)
    2. Finnhub (FREE - 60/min, has P/E)
    3. Alpha Vantage (LIMITED - 5/min)
    4. Marketstack (PAID - save for backup only)
    
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
        
        # PRIMARY: Yahoo Finance (FREE - unlimited requests)
        try:
            logger.info(f"Trying Yahoo Finance for {ticker}")
            
            url = f"https://query1.finance.yahoo.com/v10/finance/quoteSummary/{ticker.upper()}?modules=price,financialData,defaultKeyStatistics,summaryDetail"
            headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                
                if 'quoteSummary' in data and 'result' in data['quoteSummary'] and len(data['quoteSummary']['result']) > 0:
                    result = data['quoteSummary']['result'][0]
                    
                    # Price
                    if 'price' in result and 'regularMarketPrice' in result['price']:
                        try:
                            price_val = result['price']['regularMarketPrice']['raw']
                            metrics['price'] = round(float(price_val), 2)
                            logger.info(f"{ticker} price: ${metrics['price']}")
                        except Exception as e:
                            logger.debug(f"Price extraction failed: {e}")
                    
                    # P/E Ratio
                    if 'summaryDetail' in result and 'trailingPE' in result['summaryDetail']:
                        try:
                            pe = result['summaryDetail']['trailingPE']['raw']
                            if pe and pe > 0:
                                metrics['pe_ratio'] = round(float(pe), 2)
                                logger.info(f"{ticker} P/E: {metrics['pe_ratio']}")
                        except:
                            pass
                    
                    # 52-week high/low
                    if 'summaryDetail' in result:
                        detail = result['summaryDetail']
                        
                        if 'fiftyTwoWeekHigh' in detail:
                            try:
                                metrics['52_week_high'] = round(float(detail['fiftyTwoWeekHigh']['raw']), 2)
                            except:
                                pass
                        
                        if 'fiftyTwoWeekLow' in detail:
                            try:
                                metrics['52_week_low'] = round(float(detail['fiftyTwoWeekLow']['raw']), 2)
                            except:
                                pass
                        
                        # Market cap
                        if 'marketCap' in detail:
                            try:
                                market_cap = detail['marketCap']['raw']
                                if market_cap >= 1e9:
                                    metrics['market_cap'] = f"${market_cap/1e9:.1f}B"
                                elif market_cap >= 1e6:
                                    metrics['market_cap'] = f"${market_cap/1e6:.1f}M"
                            except:
                                pass
                        
                        # Dividend yield
                        if 'trailingAnnualDividendYield' in detail:
                            try:
                                div = detail['trailingAnnualDividendYield']['raw']
                                if div and div > 0:
                                    metrics['dividend_yield'] = round(float(div) * 100, 2)
                            except:
                                pass
                    
                    if metrics['price'] != 'N/A':
                        metrics['data_source'] = 'Yahoo Finance'
                        logger.info(f"Yahoo Finance success for {ticker}")
                        return metrics
        
        except Exception as yf_err:
            logger.debug(f"Yahoo Finance failed: {yf_err}")
        
        # SECONDARY: Finnhub (FREE - 60 calls/min, has better P/E data)
        if FINNHUB_API_KEY:
            try:
                logger.info(f"Trying Finnhub for {ticker}")
                
                quote_url = f"{FINNHUB_BASE_URL}/quote?symbol={ticker.upper()}&token={FINNHUB_API_KEY}"
                quote_response = requests.get(quote_url, timeout=5)
                
                if quote_response.status_code == 200:
                    quote_data = quote_response.json()
                    
                    if quote_data.get('c'):
                        metrics['price'] = round(float(quote_data.get('c')), 2)
                    if quote_data.get('h52'):
                        metrics['52_week_high'] = round(float(quote_data.get('h52')), 2)
                    if quote_data.get('l52'):
                        metrics['52_week_low'] = round(float(quote_data.get('l52')), 2)
                    if quote_data.get('pe'):
                        try:
                            metrics['pe_ratio'] = round(float(quote_data.get('pe')), 2)
                        except:
                            pass
                    
                    if metrics['price'] != 'N/A':
                        metrics['data_source'] = 'Finnhub'
                        logger.info(f"Finnhub success for {ticker}")
                        return metrics
            
            except Exception as fh_err:
                logger.debug(f"Finnhub failed: {fh_err}")
        
        # TERTIARY: Alpha Vantage (LIMITED - 5 calls/min, save for company info)
        if ALPHA_VANTAGE_API_KEY != "demo":
            try:
                logger.info(f"Trying Alpha Vantage for {ticker}")
                
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
                    
                    if metrics['price'] != 'N/A':
                        metrics['data_source'] = 'Alpha Vantage'
                        logger.info(f"Alpha Vantage success for {ticker}")
                        return metrics
            
            except Exception as av_err:
                logger.debug(f"Alpha Vantage failed: {av_err}")
        
        # QUATERNARY: Marketstack (PAID - save for backup only)
        if MARKETSTACK_API_KEY and metrics['price'] == 'N/A':
            try:
                logger.info(f"Trying Marketstack for {ticker} (using paid quota)")
                
                url = f"{MARKETSTACK_BASE_URL}/eod"
                params = {
                    'access_key': MARKETSTACK_API_KEY,
                    'symbols': ticker.upper(),
                    'limit': 1
                }
                response = requests.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                
                if data.get('data') and len(data['data']) > 0:
                    result = data['data'][0]
                    
                    if result.get('close'):
                        metrics['price'] = round(float(result.get('close')), 2)
                    if result.get('high'):
                        metrics['52_week_high'] = round(float(result.get('high')), 2)
                    if result.get('low'):
                        metrics['52_week_low'] = round(float(result.get('low')), 2)
                    
                    if metrics['price'] != 'N/A':
                        metrics['data_source'] = 'Marketstack'
                        logger.info(f"Marketstack success for {ticker}")
                        return metrics
            
            except Exception as ms_err:
                logger.debug(f"Marketstack failed: {ms_err}")
        
        logger.error(f"All data sources failed for {ticker}")
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
    Fetch stock fundamentals - prioritize free sources
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary with stock data
    """
    try:
        # Use Alpha Vantage only if configured
        if ALPHA_VANTAGE_API_KEY != "demo":
            params = {
                'function': 'OVERVIEW',
                'symbol': ticker,
                'apikey': ALPHA_VANTAGE_API_KEY
            }
            response = requests.get(ALPHA_VANTAGE_BASE_URL, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        
        return {}
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
    
    sector = fundamentals.get('Sector', 'the technology sector')
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
        st.info("No source articles available for this recommendation")
    
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
        
        category_data = {
            'Large-Cap': 0,
            'Mid-Cap': 0,
            'Small-Cap': 0,
            'ETF': 0
        }
        
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
            
            st.bar_chart(
                category_df.set_index('Category'),
                use_container_width=True,
                height=300
            )
        
        with col2:
            st.markdown("**Top 10 Mentioned Tickers**")
            top_tickers_df = pd.DataFrame(ticker_mentions).sort_values('Mentions', ascending=False).head(10)
            
            st.bar_chart(
                top_tickers_df.set_index('Ticker')['Mentions'],
                use_container_width=True,
                height=300
            )
        
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
    st.markdown("*Data sources prioritized: Yahoo Finance (free) → Finnhub (free) → Alpha Vantage (limited) → Marketstack (paid backup only)*")
    
    if articles:
        mentions = extract_stock_mentions(articles)
        
        metrics_data = []
        progress_bar = st.progress(0)
        total_tickers = len(mentions)
        
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
            
            progress_bar.progress((idx + 1) / total_tickers)
        
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
            **API Usage Optimization:**
            
            ✅ **Yahoo Finance (PRIMARY - UNLIMITED)**
            - Price, P/E Ratio, 52W High/Low, Market Cap, Dividend Yield
            - No API key required, unlimited requests
            
            ✅ **Finnhub (SECONDARY - 60 calls/min free)**
            - Fallback for price and P/E data if Yahoo fails
            - Get key: https://finnhub.io/
            
            ✅ **Alpha Vantage (TERTIARY - 5 calls/min limited)**
            - Only used for company fundamentals (OVERVIEW function)
            - Get key: https://www.alphavantage.co/
            
            ⭐ **Marketstack (BACKUP ONLY - PAID)**
            - Only used if all free sources fail
            - Free tier: 100 requests/month (NOT recommended for daily use)
            - Get key: https://marketstack.com/
            
            ✅ **FRED (MACRO - UNLIMITED)**
            - S&P 500 CAPE ratio for market-wide valuation
            - Get key: https://fred.stlouisfed.org/
            
            **Recommendation:** Start with just Yahoo Finance + optional Finnhub. 
            Add others only if needed. Yahoo Finance is the most reliable and unlimited!
            """)
        else:
            st.info("Financial metrics data not available. Check your network connection.")
    
    st.markdown("---")
    
    if recommendations:
        st.markdown("### 🎯 Recommendations Summary")
        
        rec_df = pd.DataFrame([
            {
                'Ticker': r['ticker'],
                'Company': r['company_name'],
                'Category': r['category'],
                'Confidence': f"{r['confidence_score']:.0%}",
                'Sector': r['sector']
            }
            for r in recommendations
        ])
        
        st.dataframe(
            rec_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Ticker': st.column_config.TextColumn('Symbol'),
                'Company': st.column_config.TextColumn('Company Name'),
                'Category': st.column_config.TextColumn('Market Cap'),
                'Confidence': st.column_config.TextColumn('Confidence Level'),
                'Sector': st.column_config.TextColumn('Sector')
            }
        )

def main():
    """Main Streamlit application"""
    
    init_session()
    
    st.title("📈 Stock Recommendation Engine")
    st.markdown("AI-powered daily stock recommendations | Optimized for free APIs (Yahoo Finance first, minimal paid API usage)")
    
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
        
        st.markdown("### 🧪 Debug Tools")
        test_ticker = st.text_input("Test ticker (e.g., AAPL):", value="AAPL")
        if st.button("📍 Test Single Ticker", use_container_width=True):
            st.info(f"Testing {test_ticker}...")
            test_metrics = get_enhanced_metrics(test_ticker)
            st.json(test_metrics)
        
        if st.button("🗑️ Clear Cache", use_container_width=True):
            clear_cache()
            st.success("Cache cleared!")
        
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
            1. Crawls financial news from 7 major sources daily
            2. Analyzes coverage across all market cap categories
            3. Extracts stock & ETF mentions
            4. Analyzes fundamentals using free APIs
            5. Generates recommendations with source links
            
            **API Strategy:** Minimizes paid API usage by prioritizing Yahoo Finance
            """)
        
        with st.expander("🔑 API Configuration"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("AI Engine")
                if HF_API_KEY:
                    st.success("✅ Hugging Face configured")
                else:
                    st.warning("⚠️ Template mode")
            
            with col2:
                st.subheader("Financial Data")
                st.success("✅ Yahoo Finance (FREE - always available)")
                if FINNHUB_API_KEY:
                    st.success("✅ Finnhub (backup)")
                if ALPHA_VANTAGE_API_KEY != "demo":
                    st.success("✅ Alpha Vantage (company info)")
                if MARKETSTACK_API_KEY:
                    st.success("✅ Marketstack (emergency backup)")
            
            st.divider()
            
            st.markdown("""
            **API Priority (Left to Right = Used First):**
            
            1️⃣ **Yahoo Finance** (RECOMMENDED - START HERE)
               - ✅ Unlimited requests, no key needed
               - ✅ Gets price, P/E, 52W, market cap, dividends
               - 💰 Cost: $0/month
            
            2️⃣ **Finnhub** (OPTIONAL - FALLBACK)
               - ✅ Free tier: 60 calls/minute
               - Get key: https://finnhub.io/
               - 💰 Cost: $0 (free tier)
            
            3️⃣ **Alpha Vantage** (OPTIONAL - COMPANY INFO ONLY)
               - ✅ Free tier: 5 calls/minute (don't use for quotes!)
               - Get key: https://www.alphavantage.co/api/
               - 💰 Cost: $0 (free tier)
            
            4️⃣ **Marketstack** (BACKUP ONLY - AVOID)
               - ⚠️ Limited to 100 requests/month (not recommended)
               - Get key: https://marketstack.com/
               - 💰 Cost: $0 free or $99+/month paid
            
            5️⃣ **FRED** (OPTIONAL - MACRO DATA)
               - ✅ Unlimited free requests
               - Get key: https://fred.stlouisfed.org/
               - 💰 Cost: $0/month
            
            **Recommended Setup:**
            - Just use Yahoo Finance (it has everything!)
            - Add Finnhub only if Yahoo fails
            - Skip Marketstack entirely - you don't need it!
            """)
    
    if st.session_state.last_crawl:
        tab1, tab2, tab3 = st.tabs(["📋 Recommendations", "📰 Crawl Data", "📊 Analytics"])
        
        with tab1:
            if st.session_state.recommendations:
                st.subheader(f"Top {len(st.session_state.recommendations)} Stock Recommendations")
                for i, rec in enumerate(st.session_state.recommendations[:num_recs], 1):
                    st.markdown(f"### #{i} - {rec['company_name']} ({rec['ticker']})")
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
                    hide_index=True,
                    column_config={
                        'source': st.column_config.TextColumn('News Source'),
                        'title': st.column_config.TextColumn('Article Title'),
                        'published': st.column_config.TextColumn('Published Date')
                    }
                )
        
        with tab3:
            render_analytics_dashboard(
                st.session_state.crawled_articles,
                st.session_state.recommendations
            )
    else:
        col1, col2 = st.columns(2)
        with col1:
            st.info("👈 Start by running the web crawler in the sidebar")
        with col2:
            st.info("Then generate recommendations based on crawled data")

if __name__ == "__main__":
    main()
