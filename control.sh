#!/bin/bash

# Forex Signal Bot Control Script (Python Version with Deep Analysis)

cd /root/.openclaw/workspace/forex-signal-bot

case "$1" in
  start)
    echo "🚀 Running bot with deep analysis..."
    python3 forex_bot_v2.py
    ;;
  status)
    echo "📊 Python Bot Status:"
    crontab -l | grep forex
    echo ""
    echo "📝 Recent Logs:"
    tail -30 /var/log/forex-bot.log 2>/dev/null || echo "No logs yet"
    ;;
  logs)
    echo "📝 Live Log Stream (Ctrl+C to exit):"
    tail -f /var/log/forex-bot.log 2>/dev/null || cat /var/log/forex-bot.log
    ;;
  test)
    echo "🧪 Sending test signal..."
    python3 -c "
import asyncio
from forex_bot_v2 import ForexAnalysisEngine, TelegramNotifier

async def test():
    engine = ForexAnalysisEngine()
    notifier = TelegramNotifier('8053803318:AAGvQSpZyRl8IzlkWUSEwg23qaYCULTrfwU', '6363282701')
    
    signals = await engine.run_analysis()
    if signals:
        await notifier.send_signal(signals[0])
        print('✅ Test signal sent!')
    else:
        print('No signals generated')

asyncio.run(test())
"
    ;;
  old-bot)
    echo "🔄 Running old Node.js bot..."
    node bot.js
    ;;
  stop)
    echo "🛑 To stop auto-run, run: crontab -e"
    echo "Then delete the forex line."
    ;;
  *)
    echo "Forex Signal Bot Control (Python v2)"
    echo ""
    echo "Usage: $0 {start|status|logs|test|stop|old-bot}"
    echo ""
    echo "Commands:"
    echo "  start    - Run new Python bot with deep analysis"
    echo "  status   - Show cron status and logs"
    echo "  logs     - View live log stream"
    echo "  test     - Send test signal to Telegram"
    echo "  old-bot  - Run old Node.js bot"
    echo "  stop     - Show how to disable auto-run"
    echo ""
    ;;
esac
