#!/usr/bin/env python3
"""
Forex News Alert Bot
Monitors real-time forex news and sends alerts to Telegram
"""

import asyncio
import aiohttp
import logging
from datetime import datetime
from typing import Dict, List
import os

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler('/var/log/forex-news.log'), logging.StreamHandler()])
logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv('TELEGRAM_TOKEN', '8053803318:AAGvQSpZyRl8IzlkWUSEwg23qaYCULTrfwU')
TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '6363282701')

# API Keys
NEWS_DATA_IO_API_KEY = os.getenv('NEWS_DATA_IO_API_KEY', 'pub_926e8c42b2cd48bdb569e4237f60fd67')
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY', 'QBCMUZBFP9I45Y9P')

# Keywords to monitor
FOREX_KEYWORDS = [
    'forex', 'EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD', 'USDCAD',
    'Federal Reserve', 'FED', 'ECB', 'European Central Bank', 'BOJ', 'Bank of Japan',
    'BOE', 'Bank of England', 'Interest Rate', 'FOMC', 'CPI', 'Inflation',
    'GDP', 'Non-farm Payrolls', 'NFP', 'Retail Sales', 'Unemployment',
    'Central Bank', 'Monetary Policy', 'Rate Hike', 'Rate Cut', 'Quantitative Easing'
]

# Track sent news to avoid duplicates
sent_news = set()

async def fetch_newsdata_io_market() -> List[Dict]:
    """Fetch from NewsData.io Market API"""
    news = []
    try:
        async with aiohttp.ClientSession() as session:
            keywords = ",".join(FOREX_KEYWORDS[:10])
            url = f"https://newsdata.io/api/1/market?apikey={NEWS_DATA_IO_API_KEY}&q={keywords}"
            async with session.get(url) as resp:
                data = await resp.json()
                if data.get('status') == 'success':
                    for item in data.get('results', []):
                        news.append({
                            'title': item.get('title', '') or '',
                            'source': item.get('source_id', 'NewsData.io'),
                            'published': item.get('pubDate', ''),
                            'link': item.get('link', ''),
                            'category': ', '.join(item.get('category', [])),
                        })
    except Exception as e:
        logger.error(f"NewsData.io error: {e}")
    return news

async def fetch_alpha_vantage_news() -> List[Dict]:
    """Fetch from Alpha Vantage News Sentiment"""
    news = []
    try:
        async with aiohttp.ClientSession() as session:
            url = f"https://www.alphavantage.co/query?function=NEWS_SENTIMENT&topics=financial_markets,economy_macro&limit=20&apikey={ALPHA_VANTAGE_API_KEY}"
            async with session.get(url) as resp:
                data = await resp.json()
                if 'feed' in data:
                    for item in data['feed']:
                        news.append({
                            'title': item.get('title', '') or '',
                            'source': item.get('source', 'Alpha Vantage'),
                            'published': item.get('time_published', ''),
                            'link': item.get('url', ''),
                            'sentiment': item.get('overall_sentiment_score', 0),
                        })
    except Exception as e:
        logger.error(f"Alpha Vantage error: {e}")
    return news

async def fetch_forex_factory() -> List[Dict]:
    """Fetch from Forex Factory Economic Calendar"""
    news = []
    try:
        async with aiohttp.ClientSession() as session:
            url = "https://www.forexfactory.com/api/calendar"
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    for event in data.get('events', [])[:10]:
                        impact = event.get('impact', {}).get('name', 'Medium')
                        if impact in ['High', 'Medium']:  # Only high/medium impact
                            news.append({
                                'title': event.get('name', 'Economic Event'),
                                'source': 'Forex Factory',
                                'published': event.get('date', ''),
                                'impact': impact,
                                'country': event.get('country', ''),
                            })
    except Exception as e:
        logger.error(f"Forex Factory error: {e}")
    return news

def filter_forex_news(news_list: List[Dict]) -> List[Dict]:
    """Filter to only forex-related news"""
    filtered = []
    for item in news_list:
        title = item.get('title', '').lower()
        for keyword in FOREX_KEYWORDS:
            if keyword.lower() in title:
                filtered.append(item)
                break
    return filtered

def create_news_hash(news: Dict) -> str:
    """Create unique hash for news item"""
    return hash(news.get('title', '') + news.get('source', ''))

async def send_telegram_alert(news: Dict):
    """Send news alert to Telegram"""
    emoji = '🔴' if news.get('impact') == 'High' or news.get('sentiment', 0) > 0.5 else '🟡'
    emoji = '🟢' if news.get('sentiment', 0) < -0.5 else emoji
    
    sentiment_emoji = '📈' if news.get('sentiment', 0) > 0 else '📉' if news.get('sentiment', 0) < 0 else '⚖️'
    
    message = f"""
{emoji} *FOREX NEWS ALERT*

📰 *{news.get('title', 'No Title')}*

📰 *Source:* {news.get('source', 'Unknown')}
{sentiment_emoji} *Sentiment:* {news.get('sentiment', 'N/A')}
🏷️ *Category:* {news.get('category', 'General')}
🔗 *Link:* {news.get('link', 'N/A')}

🕐 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
                json={
                    'chat_id': TELEGRAM_CHAT_ID,
                    'text': message,
                    'parse_mode': 'Markdown',
                    'disable_web_page_preview': False
                }
            ) as resp:
                if resp.status == 200:
                    logger.info(f"✅ News alert sent: {news.get('title', '')[:50]}...")
                    return True
                else:
                    logger.error(f"Telegram error: {await resp.text()}")
    except Exception as e:
        logger.error(f"Error sending alert: {e}")
    return False

async def check_and_send_news():
    """Main function to check and send forex news"""
    logger.info("🔍 Checking for forex news...")
    
    all_news = []
    
    # Fetch from all sources
    newsdata_news = await fetch_newsdata_io_market()
    alpha_news = await fetch_alpha_vantage_news()
    factory_news = await fetch_forex_factory()
    
    all_news.extend(newsdata_news)
    all_news.extend(alpha_news)
    all_news.extend(factory_news)
    
    # Filter forex-related
    forex_news = filter_forex_news(all_news)
    
    sent_count = 0
    for news in forex_news[:15]:  # Limit to 15 news items
        news_hash = create_news_hash(news)
        if news_hash not in sent_news:
            sent_news.add(news_hash)
            if await send_telegram_alert(news):
                sent_count += 1
            await asyncio.sleep(1)  # Rate limiting
    
    logger.info(f"✅ Sent {sent_count} news alerts")
    return sent_count

async def main():
    """Main entry point"""
    logger.info("🚀 Starting Forex News Alert Bot...")
    count = await check_and_send_news()
    logger.info(f"✅ Complete! Sent {count} news alerts")

if __name__ == '__main__':
    asyncio.run(main())
