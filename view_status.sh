#!/bin/bash

# Quick status viewer
clear
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║       CRYPTO TRADING BOT - QUICK STATUS                   ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

if [ -f trading_status.txt ]; then
    cat trading_status.txt
else
    echo "❌ Status file not found. Is the bot running?"
    echo ""
    echo "Start with: python paper_trade_desktop.py"
fi

echo ""
echo "Press Ctrl+C to exit, or run: watch -n 10 ./view_status.sh"
