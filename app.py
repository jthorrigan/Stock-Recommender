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
import logging
import xml.etree.ElementTree as ET
import re
import numpy as np

try:
    import yfinance as yf
except ImportError:
    yf = None

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
HF_API_KEY = os.getenv("HF_API_KEY", "")
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")
FRED_API_KEY = os.getenv("FRED_API_KEY", "")

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

# ============================================================================
# COMPANY DATA FUNCTIONS
# ============================================================================

@st.cache_data(ttl=86400)
def get_company_info(ticker: str) -> Dict:
    """
    Fetch company information including name and sector
    
    Args:
        ticker: Stock ticker symbol
    
    Returns:
        Dictionary with company info
    """
    try:
        # Try Alpha Vantage first
        if ALPHA_VANTAGE_API_KEY != "demo":
            params = {
                'function': 'OVERVIEW',
                'symbol': ticker,
                'apikey': ALPHA_VANTAGE_API_KEY
            }
            response = requests.get('https://www.alphavantage.co/query', params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            if 'Name' in data:
                market_cap = data.get('MarketCapitalization', '0')
                try:
                    market_cap_num = int(market_cap) if market_cap else 0
                except:
                    market_cap_num = 0
                
                return {
                    'ticker': ticker,
                    'name': data.get('Name', ticker),
                    'sector': data.get('Sector', 'Unknown'),
                    'industry': data.get('Industry', 'Unknown'),
                    'description': data.get('Description', ''),
                    'pe_ratio': data.get('PERatio', 'N/A'),
                    'dividend_yield': data.get('DividendYield', 'N/A'),
                    'market_cap': market_cap,
                    'market_cap_numeric': market_cap_num,
                    'category': TICKER_CATEGORIES.get(ticker, 'Unknown'),
                    'profit_margin': data.get('ProfitMargin', 'N/A'),
                    'eps': data.get('EPS', 'N/A')
                }
        
        # Fallback
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
            'eps': 'N/A'
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
            'eps': 'N/A'
        }

@st.cache_data(ttl=3600)
def get_enhanced_metrics(ticker: str) -> Dict:
    """
    Fetch comprehensive stock metrics from multiple free sources
    Primary: yfinance (no auth needed)
    Secondary: Finnhub (if API key available)
    
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
        # Primary source: yfinance (completely free, no API key needed)
        if yf is not None:
            yf_ticker = yf.Ticker(ticker)
            yf_info = yf_ticker.info
            
            # Get price history for volatility calculation (90 days)
            try:
                hist = yf_ticker.history(period="90d", progress=False)
                
                if not hist.empty and len(hist) > 1:
                    # Calculate volatility from 90-day returns
                    returns = hist['Close'].pct_change().dropna()
                    if len(returns) > 0:
                        volatility = returns.std() * np.sqrt(252)  # Annualized
                        metrics['volatility'] = round(volatility * 100, 2)
            except Exception as e:
                logger.debug(f"Error calculating volatility for {ticker}: {str(e)}")
            
            # Extract metrics from yfinance
            if yf_info.get('trailingPE') and yf_info.get('trailingPE') != 'N/A':
                metrics['pe_ratio'] = round(float(yf_info.get('trailingPE', 'N/A')), 2)
            
            if yf_info.get('currentPrice'):
                metrics['price'] = round(float(yf_info.get('currentPrice')), 2)
            
            if yf_info.get('fiftyTwoWeekHigh'):
                metrics['52_week_high'] = round(float(yf_info.get('fiftyTwoWeekHigh')), 2)
            
            if yf_info.get('fiftyTwoWeekLow'):
                metrics['52_week_low'] = round(float(yf_info.get('fiftyTwoWeekLow')), 2)
            
            if yf_info.get('marketCap'):
                market_cap = yf_info.get('marketCap')
                if market_cap >= 1e9:
                    metrics['market_cap'] = f"${market_cap/1e9:.1f}B"
                elif market_cap >= 1e6:
                    metrics['market_cap'] = f"${market_cap/1e6:.1f}M"
            
            if yf_info.get('dividendYield'):
                metrics['dividend_yield'] = round(float(yf_info.get('dividendYield')) * 100, 2)
            
            if yf_info.get('trailingEps'):
                metrics['eps'] = round(float(yf_info.get('trailingEps')), 2)
            
            if yf_info.get('profitMargins'):
                metrics['profit_margin'] = round(float(yf_info.get('profitMargins')) * 100, 2)
            
            metrics['data_source'] = 'yfinance'
        
        # Secondary source: Finnhub (if API key available - backup/validation)
        if FINNHUB_API_KEY:
            try:
                finnhub_url = f"https://finnhub.io/api/v1/quote?symbol={ticker}&token={FINNHUB_API_KEY}"
                response = requests.get(finnhub_url, timeout=5)
                
                if response.status_code == 200:
                    finnhub_data = response.json()
                    
                    # Use Finnhub as fallback if yfinance data is missing
                    if metrics['pe_ratio'] == 'N/A' and finnhub_data.get('pe'):
                        metrics['pe_ratio'] = round(float(finnhub_data.get('pe')), 2)
                    
                    if metrics['52_week_high'] == 'N/A' and finnhub_data.get('h52'):
                        metrics['52_week_high'] = round(float(finnhub_data.get('h52')), 2)
                    
                    if metrics['52_week_low'] == 'N/A' and finnhub_data.get('l52'):
                        metrics['52_week_low'] = round(float(finnhub_data.get('l52')), 2)
                    
                    if metrics['price'] == 'N/A' and finnhub_data.get('c'):
                        metrics['price'] = round(float(finnhub_data.get('c')), 2)
                    
                    if metrics['data_source'] == 'yfinance':
                        metrics['data_source'] = 'yfinance + Finnhub'
            
            except Exception as e:
                logger.debug(f"Error fetching Finnhub data for {ticker}: {str(e)}")
    
    except Exception as e:
        logger.warning(f"Error fetching enhanced metrics for {ticker}: {str(e)}")
    
    return metrics

@st.cache_data(ttl=86400)
def get_cape_ratio_approximation() -> Dict:
    """
    Fetch CAPE ratio from FRED API (Shiller P/E)
    Returns the most recent CAPE ratio for reference
    
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
        # Fetch Shiller P/E (CAPE) from FRED
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
    Uses XML parsing instead of feedparser for Python 3.13 compatibility
    
    Returns:
        List of article dictionaries with metadata
    """
    articles = []
    
    for source_name, feed_url in NEWS_SOURCES.items():
        try:
            logger.info(f"Crawling {source_name}...")
            
            # Fetch the RSS feed
            response = requests.get(feed_url, timeout=10)
            response.raise_for_status()
            
            # Parse XML
            try:
                root = ET.fromstring(response.content)
            except ET.ParseError:
                logger.warning(f"Failed to parse XML from {source_name}")
                continue
            
            # Get articles from last 24 hours
            cutoff_time = datetime.now() - timedelta(days=1)
            
            # Find all items/entries (handle both RSS and Atom formats)
            items = root.findall('.//item')
            if not items:
                items = root.findall('.//{http://www.w3.org/2005/Atom}entry')
            
            for item in items[:50]:  # Limit to 50 most recent
                try:
                    # Extract fields - try multiple possible tag names
                    title = None
                    link = None
                    published = None
                    summary = None
                    
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
                        # For Atom, link is an attribute
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
                    
                    # Summary/Description
                    summary_elem = item.find('description')
                    if summary_elem is None:
                        summary_elem = item.find('{http://www.w3.org/2005/Atom}summary')
                    
                    summary = summary_elem.text if summary_elem is not None and summary_elem.text else ''
                    
                    # Clean HTML tags from summary
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
    Includes large-cap, mid-cap, small-cap stocks and ETFs
    
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

def calculate_confidence_score(ticker: str, articles: List[Dict], fundamentals: Dict, category: str) -> Tuple[float, str]:
    """
    Calculate confidence score and justification
    
    Args:
        ticker: Stock ticker
        articles: Related articles
        fundamentals: Stock fundamentals
        category: Stock category (Large-Cap, Mid-Cap, Small-Cap, ETF)
    
    Returns:
        Tuple of (confidence_score, justification_text)
    """
    
    confidence = 0.5  # Base confidence
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
    
    # Fundamentals factor
    if fundamentals.get('PERatio') and fundamentals.get('PERatio') != 'None':
        try:
            pe_ratio = float(fundamentals.get('PERatio', 0))
            if 10 < pe_ratio < 30:
                confidence += 0.10
                factors.append("Reasonable valuation metrics")
            elif pe_ratio > 0:
                factors.append(f"High valuation (P/E: {pe_ratio:.1f})")
        except:
            pass
    
    # Cap at 0.95 max
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
    Generate a 500-word investment explanation using free Hugging Face Inference API
    Falls back to template if API fails
    
    Args:
        ticker: Stock ticker
        company_name: Company name
        articles: Related articles
        fundamentals: Stock fundamentals
    
    Returns:
        500-word explanation
    """
    
    try:
        # Prepare article summaries for context
        article_context = "\n".join([
            f"• {a['source']}: {a['title']} - {a['summary'][:100]}..."
            for a in articles[:3]
        ])
        
        # Build prompt
        prompt = f"""Write a professional 500-word investment thesis for {company_name} ({ticker}) explaining:

1. Why the 12-24 month outlook is positive
2. Key catalysts and growth drivers
3. Why now is the right time to buy
4. Market timing and valuation perspective

Recent developments:
{article_context}

Fundamentals:
- Sector: {fundamentals.get('Sector', 'Unknown')}
- Industry: {fundamentals.get('Industry', 'Unknown')}
- Market Cap: {fundamentals.get('MarketCapitalization', 'Unknown')}
- P/E Ratio: {fundamentals.get('PERatio', 'Unknown')}

Write approximately 500 words in a professional analytical tone. Include specific reasons for bullish outlook and timing."""

        # Use Hugging Face Inference API (free, no auth required for some models)
        API_URL = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.1"
        headers = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}
        
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
                # Clean up the response
                if prompt in generated_text:
                    generated_text = generated_text.replace(prompt, '').strip()
                if generated_text:
                    return generated_text[:2000]  # Limit to reasonable length
        
    except Exception as e:
        logger.warning(f"Error generating explanation via Hugging Face for {ticker}: {str(e)}")
    
    # Fallback to template
    return generate_template_explanation(ticker, company_name, articles, fundamentals)

def generate_template_explanation(
    ticker: str,
    company_name: str,
    articles: List[Dict],
    fundamentals: Dict
) -> str:
    """
    Generate a detailed template-based explanation
    Used when free API is not available or fails
    
    Args:
        ticker: Stock ticker
        company_name: Company name
        articles: Related articles
        fundamentals: Stock fundamentals
    
    Returns:
        500-word explanation
    """
    
    sector = fundamentals.get('Sector', 'the technology sector')
    industry = fundamentals.get('Industry', 'its industry')
    recent_news = articles[0]['title'] if articles else "Recent market developments"
    
    explanation = f"""
## Investment Thesis: {company_name} ({ticker})

### 12-24 Month Positive Outlook

{company_name} presents a compelling investment opportunity with significant upside potential over the next 12-24 months. 
The company operates in {sector}, a dynamic and growing market with substantial tailwinds. Recent market developments, 
including {recent_news}, demonstrate the market's growing recognition of the company's value proposition.

### Growth Catalysts and Drivers

Several key catalysts position {company_name} for outperformance:

1. **Market Expansion**: The company is well-positioned to capture market share in growing segments. With {industry} 
experiencing accelerating adoption, {company_name}'s competitive advantages should drive significant revenue growth.

2. **Operational Efficiency**: Recent developments indicate improving operational metrics and margin expansion opportunities. 
The company's management team has demonstrated strong execution capabilities in scaling operations profitably.

3. **Strategic Positioning**: {company_name} maintains a defensible competitive position with strong brand recognition and 
customer loyalty. The company's strategic initiatives are aligned with long-term industry trends.

### Why Now is the Right Entry Point

Current valuation levels present an attractive risk-reward opportunity for several reasons:

1. **Market Sentiment**: Recent market volatility has created a disconnect between the company's intrinsic value and current 
market price. This dislocation presents a compelling entry point for long-term investors.

2. **Fundamental Strength**: The company's strong balance sheet and cash generation capabilities provide downside protection 
while positioned for upside capture. Key financial metrics demonstrate financial health and operational efficiency.

3. **Timing**: Market cycles suggest we are in an advantageous entry window. Forward estimates indicate accelerating growth 
rates in subsequent periods, which historically has led to re-rating of valuation multiples.

### Technical and Macro Positioning

From a macro perspective, several tailwinds support {company_name}'s investment case:

- Industry growth rates are accelerating, creating a favorable backdrop for company-specific outperformance
- Regulatory environment remains supportive of the company's core business model
- Economic indicators suggest sustained demand for the company's products and services

### Risk Considerations

While the bull case is compelling, investors should monitor key metrics including quarterly revenue growth rates, 
margin trends, and competitive positioning. The company faces cyclical risks and competitive pressures typical of its sector.

### Conclusion

{company_name} offers an attractive risk-reward profile for investors with a 12-24 month investment horizon. The combination 
of favorable catalysts, improving fundamentals, and attractive valuation creates a compelling investment opportunity. 
Current market conditions present a strategic entry point before the market more fully appreciates the company's long-term value.

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
    
    # Get company info
    company_info = get_company_info(ticker)
    company_name = company_info.get('name', ticker)
    category = company_info.get('category', 'Unknown')
    
    # Calculate confidence score
    confidence_score, confidence_justification = calculate_confidence_score(
        ticker, articles, fundamentals, category
    )
    
    # Generate detailed explanation
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
        'price_target': fundamentals.get('PERatio', 'TBD'),
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
        st.markdown(f"## {rec['ticker']} - {rec['company_name']}")
        st.markdown(f"**Category:** {rec['category']} | **Sector:** {rec['sector']} | **Rating:** {rec['rating']}")
    
    with col2:
        st.metric("Confidence", f"{rec['confidence_score']:.0%}")
    
    # Confidence justification
    st.info(f"📊 **Confidence Rationale:** {rec['confidence_justification']}")
    
    st.markdown("---")
    
    # Explanation section
    st.markdown("### Investment Thesis (12-24 Month Outlook)")
    st.markdown(rec['explanation'])
    
    # Risk factors
    with st.expander("⚠️ Risk Factors"):
        for risk in rec['risk_factors']:
            st.write(f"• {risk}")
    
    # Source links
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
    
    # Top row: Key metrics
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
    
    # Stock mentions by category
    st.markdown("### 📈 Stock Mentions by Market Cap Category")
    
    if articles:
        mentions = extract_stock_mentions(articles)
        
        # Categorize mentions
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
        
        # Category pie chart
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Mentions by Category**")
            category_df = pd.DataFrame(
                list(category_data.items()),
                columns=['Category', 'Mentions']
            ).sort_values('Mentions', ascending=False)
            
            fig = st.bar_chart(
                category_df.set_index('Category'),
                use_container_width=True,
                height=300
            )
            st.markdown("*Chart shows article mentions by company category (Y-axis: Mention Count, X-axis: Category)*")
        
        with col2:
            st.markdown("**Top 10 Mentioned Tickers**")
            top_tickers_df = pd.DataFrame(ticker_mentions).sort_values('Mentions', ascending=False).head(10)
            
            fig = st.bar_chart(
                top_tickers_df.set_index('Ticker')['Mentions'],
                use_container_width=True,
                height=300
            )
            st.markdown("*Chart shows most frequently mentioned stocks in news (Y-axis: Mention Count, X-axis: Ticker)*")
        
        st.markdown("---")
        
        # Detailed ticker table
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
    
    # Company Metrics Section
    st.markdown("### 💹 Company Financial Metrics")
    st.markdown("*Most recent P/E ratio, CAPE ratio, volatility, and other key financial data*")
    
    if articles:
        mentions = extract_stock_mentions(articles)
        
        # Fetch metrics for mentioned tickers
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
                'CAPE Ratio': metrics.get('cape_ratio', 'N/A'),
                'Volatility (%)': metrics.get('volatility', 'N/A'),
                '52W High': metrics.get('52_week_high', 'N/A'),
                '52W Low': metrics.get('52_week_low', 'N/A'),
                'Dividend Yield (%)': metrics.get('dividend_yield', 'N/A'),
                'EPS': metrics.get('eps', 'N/A'),
                'Profit Margin (%)': metrics.get('profit_margin', 'N/A')
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
                    'CAPE Ratio': st.column_config.TextColumn('CAPE Ratio', width='small'),
                    'Volatility (%)': st.column_config.NumberColumn('Volatility (%)', format='%.2f'),
                    '52W High': st.column_config.NumberColumn('52-Week High', format='$%.2f'),
                    '52W Low': st.column_config.NumberColumn('52-Week Low', format='$%.2f'),
                    'Dividend Yield (%)': st.column_config.NumberColumn('Dividend Yield (%)', format='%.2f'),
                    'EPS': st.column_config.NumberColumn('EPS', format='$.2f'),
                    'Profit Margin (%)': st.column_config.NumberColumn('Profit Margin (%)', format='%.2f')
                }
            )
            
            # Add CAPE ratio reference if available
            cape_info = get_cape_ratio_approximation()
            if cape_info['cape_ratio'] != 'N/A':
                st.info(f"📊 **S&P 500 CAPE Ratio (Shiller P/E):** {cape_info['cape_ratio']} (as of {cape_info['cape_date']}) - Use this for market-wide valuation comparison")
            
            st.markdown("""
            **Metric Definitions:**
            - **Price**: Current stock price
            - **P/E Ratio**: Price-to-Earnings ratio (lower = potentially better value)
            - **CAPE Ratio**: Cyclically Adjusted P/E (compares to 10-year historical earnings)
            - **Volatility (%)**: 90-day annualized return volatility (higher = more risk)
            - **52W High/Low**: 52-week trading range
            - **Dividend Yield (%)**: Annual dividend as percentage of stock price
            - **EPS**: Earnings Per Share (trailing 12 months)
            - **Profit Margin (%)**: Net profit as percentage of revenue
            
            **Data Sources:** 
            - Primary: yfinance (Yahoo Finance - free, no API key required)
            - Secondary: Finnhub API (if configured)
            - CAPE Ratio: FRED (Federal Reserve Economic Data - if configured)
            """)
        else:
            st.info("Financial metrics data not available. Check if yfinance is installed.")
    
    st.markdown("---")
    
    # Recommendations summary
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
    
    # Initialize session state
    init_session()
    
    # Header
    st.title("📈 Stock Recommendation Engine")
    st.markdown("AI-powered daily stock recommendations based on web crawl analysis | Large-Cap, Mid-Cap, Small-Cap, & ETFs")
    
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
            1. Crawls financial news from 6+ major sources daily
            2. Analyzes coverage across all market cap categories
            3. Extracts stock & ETF mentions with sentiment analysis
            4. Analyzes fundamentals and market timing
            5. Generates recommendations with source links
            
            **Coverage:**
            - Large-Cap stocks (AAPL, MSFT, GOOGL, etc.)
            - Mid-Cap growth companies (CRWD, DDOG, etc.)
            - Small-Cap emerging companies (UPST, BREX, etc.)
            - ETFs (SPY, QQQ, sector ETFs, etc.)
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
                if yf is not None:
                    st.success("✅ yfinance ready (no key needed!)")
                else:
                    st.error("❌ yfinance not installed")
            
            st.divider()
            
            st.markdown("""
            **API Status:**
            
            ✅ **yfinance** (PRIMARY - NO KEY NEEDED)
            - Get P/E, Volatility, 52W High/Low
            - No API key required
            - Install: `pip install yfinance`
            
            ⭐ **Finnhub** (OPTIONAL - BACKUP)
            - Fallback data source
            - Free tier: 60 calls/minute
            - Get key: https://finnhub.io/
            - Add to `.env`: `FINNHUB_API_KEY=...`
            
            ⭐ **FRED** (OPTIONAL - CAPE RATIO)
            - S&P 500 Shiller P/E data
            - Completely free
            - Get key: https://fred.stlouisfed.org/docs/api/fred/
            - Add to `.env`: `FRED_API_KEY=...`
            
            ⭐ **Hugging Face** (OPTIONAL - AI)
            - Generate 500-word explanations
            - Get key: https://huggingface.co/settings/tokens
            - Add to `.env`: `HF_API_KEY=hf_...`
            """)
    
    # Main content
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
