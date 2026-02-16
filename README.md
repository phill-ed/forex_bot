# 📈 Forex Signal Bot

Real-time forex monitoring bot that combines technical analysis with news sentiment to generate trading signals.

## Features

- 📊 **Real-time forex rate monitoring** (EUR, GBP, JPY, CHF, AUD, CAD)
- 📰 **News sentiment analysis** - analyzes economic news impact
- 🎯 **Technical indicators** - RSI, MACD, Moving Averages
- 📱 **Telegram notifications** - Instant signal alerts
- ⏰ **Hourly automated scanning** - Runs every hour automatically

## Quick Setup

### 1. Install Dependencies

```bash
cd /root/.openclaw/workspace/forex-signal-bot
npm install
```

### 2. Configure Telegram Bot

**Get Telegram Bot Token:**
1. Message @BotFather on Telegram
2. Send `/newbot`
3. Follow prompts to create bot
4. Copy the HTTP API token

**Get Your Chat ID:**
1. Message @userinfobot on Telegram
2. Copy your numeric user ID

### 3. Set Environment Variables

```bash
export TELEGRAM_BOT_TOKEN="your-bot-token-here"
export TELEGRAM_CHAT_ID="your-chat-id-here"
```

Or create a `.env` file:
```bash
TELEGRAM_BOT_TOKEN=your-bot-token
TELEGRAM_CHAT_ID=your-chat-id
```

### 4. Run Manually

```bash
npm start
```

### 5. Setup Automatic Hourly Runs

```bash
# Add to crontab to run every hour
crontab -e

# Add this line:
0 * * * * cd /root/.openclaw/workspace/forex-signal-bot && node bot.js >> /var/log/forex-bot.log 2>&1
```

## Bot Output Example

```
🚀 Starting Forex Signal Bot...
📅 2026-02-16T10:00:00.000Z
📊 Fetching forex rates...
✅ Rates fetched successfully
📰 Fetching economic news...
✅ 3 news items processed
📈 Generated 2 signals
  BUY EUR/USD - 75% confidence
  SELL EUR/JPY - 70% confidence
✅ Signal sent for EUR/USD
✅ Signal sent for EUR/JPY
✅ Bot run complete
```

## Signal Format

🟢 **EUR/USD** - BUY (78%)

📊 **Technical Analysis:**
• RSI (14): 28
• Trend: BULLISH
• MACD: 0.0012

📰 **Market Sentiment:** POSITIVE

💡 **Key Reasons:**
• RSI oversold with bullish trend
• MACD bullish
• Positive news sentiment

💰 **Current Price:** 1.1865

⏰ Feb 16, 2026, 10:00 AM

## How Signals Are Generated

### Technical Analysis (70% weight)
| Indicator | Buy Signal | Sell Signal |
|-----------|------------|-------------|
| RSI (14) | < 30 (oversold) | > 70 (overbought) |
| Trend (SMA 20/50) | Price > SMA20 | Price < SMA20 |
| MACD | Positive crossover | Negative crossover |

### Sentiment Analysis (30% weight)
- Positive news sentiment → Bullish bias
- Negative news sentiment → Bearish bias
- Hawkish central bank news → Currency strength
- Dovish central bank news → Currency weakness

### Confidence Score
- Base: 50%
- RSI conditions: +15%
- MACD confirmation: +10%
- Sentiment match: +5%
- Maximum: 95%

## API Limitations

**Frankfurter API** provides:
- ✅ Free, no API key required
- ✅ Historical daily data
- ⚠️ No real-time intraday data
- ⚠️ Updates once per day

**For intraday data, consider upgrading to:**
- Alpha Vantage (free tier available)
- TradingView API
- OANDA API
- MetaTrader API

## Customization

Edit `bot.js` to change:

```javascript
// Monitor different pairs
pairs: ['EUR/USD', 'GBP/USD', 'USD/JPY', 'USD/CHF', 'AUD/USD']

// Adjust sensitivity
thresholds: {
  rsiOverbought: 70,
  rsiOversold: 30,
  minConfidence: 65  // Only send signals above 65%
}

// Change schedule
schedule: '0 * * * *'  // Every hour
```

## Files

```
forex-signal-bot/
├── bot.js           # Main bot script
├── package.json     # Dependencies
├── README.md        # This file
└── .env            # Environment variables (create manually)
```

## Disclaimer

⚠️ **Important:** This bot is for educational purposes only. Trading forex involves substantial risk of loss. Always:
- Do your own research
- Use proper risk management
- Never risk money you can't afford to lose
- Test strategies on demo accounts first

## License

MIT License - Feel free to modify and use!
