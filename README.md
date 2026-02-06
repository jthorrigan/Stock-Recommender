# Stock Recommendation Engine

An AI-powered Streamlit web app that crawls financial news, SEC filings, and company reports to recommend stocks with detailed analysis.

## Features

- **Daily Web Crawler**: Automatically crawls 6+ financial news sources daily
- **SEC Filing Analysis**: Tracks 10-K, 10-Q, and 8-K filings for companies
- **Stock Recommendations**: Generates top 5 stock picks with 500+ word explanations
- **Sourced Analysis**: Every recommendation includes hyperlinks to original sources
- **12-24 Month Outlook**: Long-term bullish thesis with timing rationale
- **Risk Assessment**: Includes risk factors and confidence scoring
- **Interactive Dashboard**: View crawl data, mentions, and detailed analysis

## Setup

### Prerequisites
- Python 3.8+
- pip or conda

### Installation

```bash
# Clone the repository
git clone <your-repo>
cd stock-recommendation-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Setup environment variables
cp .env.example .env
# Edit .env and add your API keys
