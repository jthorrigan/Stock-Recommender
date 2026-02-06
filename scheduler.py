"""
Background scheduler for daily web crawls
Run this as a separate process or use with APScheduler
"""

import schedule
import time
import logging
from datetime import datetime
from app import crawl_news_feeds, crawl_sec_filings, generate_recommendations
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

CACHE_FILE = "data/crawl_cache.json"

def save_results(articles, recommendations):
    """Save crawl results to cache"""
    results = {
        'crawled_at': datetime.now().isoformat(),
        'articles': articles,
        'recommendations': recommendations
    }
    with open(CACHE_FILE, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to cache")

def daily_crawl_job():
    """Daily crawl job"""
    logger.info("Starting daily web crawl...")
    try:
        articles = crawl_news_feeds()
        recommendations = generate_recommendations(articles)
        save_results(articles, recommendations)
        logger.info(f"Daily crawl complete: {len(articles)} articles, {len(recommendations)} recommendations")
    except Exception as e:
        logger.error(f"Daily crawl failed: {str(e)}")

def schedule_daily_crawls():
    """Schedule daily crawls at specific time"""
    # Run at 9 AM daily
    schedule.every().day.at("09:00").do(daily_crawl_job)
    
    logger.info("Scheduler started - daily crawls scheduled at 09:00")
    
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    schedule_daily_crawls()
