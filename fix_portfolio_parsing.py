#!/usr/bin/env python3
"""
Fix Portfolio Parsing Issues

This script fixes the decision parsing in the portfolio checker to properly
extract BUY/SELL/HOLD decisions from the evaluation logs.
"""

import json
import re
from pathlib import Path
from datetime import datetime
import yfinance as yf

def extract_decision(decision_text):
    """Extract BUY/SELL/HOLD from decision text."""
    if not decision_text:
        return 'HOLD'
    
    # Convert to uppercase for consistent matching
    text_upper = decision_text.upper()
    
    # Look for explicit SELL indicators
    sell_patterns = [
        r'\*\*SELL\*\*',
        r'SELL\s+\w+',  # SELL AAPL, SELL GOOGL, etc.
        r'RECOMMENDATION:\s*\*\*SELL\*\*',
        r'FINAL\s+TRANSACTION\s+PROPOSAL:\s*\*\*SELL\*\*',
        r'RECOMMENDATION:\s*SELL',
        r'ACTION:\s*SELL',
        r'DECISION:\s*SELL'
    ]
    
    # Look for explicit BUY indicators
    buy_patterns = [
        r'\*\*BUY\*\*',
        r'BUY\s+\w+',  # BUY AAPL, BUY GOOGL, etc.
        r'RECOMMENDATION:\s*\*\*BUY\*\*',
        r'FINAL\s+TRANSACTION\s+PROPOSAL:\s*\*\*BUY\*\*',
        r'RECOMMENDATION:\s*BUY',
        r'ACTION:\s*BUY',
        r'DECISION:\s*BUY'
    ]
    
    # Check for SELL first (more specific)
    for pattern in sell_patterns:
        if re.search(pattern, text_upper):
            return 'SELL'
    
    # Check for BUY
    for pattern in buy_patterns:
        if re.search(pattern, text_upper):
            return 'BUY'
    
    # Default to HOLD if no clear decision found
    return 'HOLD'

def get_current_price(ticker):
    """Get current price for a ticker."""
    try:
        stock = yf.Ticker(ticker)
        hist = stock.history(period="1d")
        if not hist.empty:
            return hist['Close'].iloc[-1]
    except:
        pass
    return None

def check_fixed_portfolio_status():
    """Check portfolio status with fixed decision parsing."""
    
    print("💰 TradingAgents Fixed Portfolio Status")
    print("=" * 50)
    print(f"🕐 Checked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Configuration
    initial_cash = 100000.0
    position_size = 20000.0  # Per position
    
    # Check eval_results directory
    results_dir = Path("eval_results")
    
    if not results_dir.exists():
        print("❌ No evaluation results found yet...")
        return
    
    # Find all tickers
    tickers = [d.name for d in results_dir.iterdir() if d.is_dir()]
    print(f"📊 Tracking tickers: {tickers}")
    print()
    
    # Reconstruct portfolio from logs
    current_cash = initial_cash
    positions = {}  # {ticker: shares}
    trade_history = []
    
    # Process all log files
    for ticker in tickers:
        ticker_dir = results_dir / ticker / "TradingAgentsStrategy_logs"
        
        if not ticker_dir.exists():
            continue
            
        log_files = list(ticker_dir.glob("full_states_log_*.json"))
        log_files.sort(key=lambda x: x.stem.split('_')[-1])
        
        print(f"📈 Processing {ticker}...")
        
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)
                
                for date, state in data.items():
                    decision_text = state.get('final_trade_decision', '')
                    
                    # Extract decision using improved parsing
                    action = extract_decision(decision_text)
                    
                    print(f"  📅 {date}: {action}")
                    
                    # Simulate trade execution
                    if action == 'BUY' and ticker not in positions:
                        # Buy position
                        current_price = get_current_price(ticker)
                        if current_price:
                            shares = int(position_size / current_price)
                            cost = shares * current_price
                            if cost <= current_cash:
                                current_cash -= cost
                                positions[ticker] = shares
                                trade_history.append({
                                    'date': date,
                                    'ticker': ticker,
                                    'action': 'BUY',
                                    'shares': shares,
                                    'price': current_price,
                                    'cost': cost
                                })
                                print(f"    ✅ Bought {shares} shares at ${current_price:.2f}")
                            else:
                                print(f"    ❌ Insufficient cash for {ticker}")
                    
                    elif action == 'SELL' and ticker in positions:
                        # Sell position
                        current_price = get_current_price(ticker)
                        if current_price:
                            shares = positions[ticker]
                            proceeds = shares * current_price
                            current_cash += proceeds
                            trade_history.append({
                                'date': date,
                                'ticker': ticker,
                                'action': 'SELL',
                                'shares': shares,
                                'price': current_price,
                                'proceeds': proceeds
                            })
                            print(f"    ✅ Sold {shares} shares at ${current_price:.2f}")
                            del positions[ticker]
                    
                    elif action == 'HOLD':
                        print(f"    ⏸️  Holding {ticker}")
                
            except Exception as e:
                print(f"    ❌ Error processing {log_file}: {e}")
    
    print()
    print("💎 PORTFOLIO VALUE CALCULATION:")
    print("-" * 40)
    print(f"💰 Cash: ${current_cash:,.2f}")
    print()
    
    if positions:
        print("📊 Current Positions:")
        total_position_value = 0
        for ticker, shares in positions.items():
            current_price = get_current_price(ticker)
            if current_price:
                value = shares * current_price
                total_position_value += value
                print(f"  {ticker}: {shares} shares @ ${current_price:.2f} = ${value:,.2f}")
            else:
                print(f"  {ticker}: {shares} shares (price unavailable)")
        
        print(f"\n📊 Total Position Value: ${total_position_value:,.2f}")
    else:
        print("📊 No current positions")
    
    total_value = current_cash + sum(
        shares * get_current_price(ticker) if get_current_price(ticker) else 0
        for ticker, shares in positions.items()
    )
    
    print()
    print("💎 EQUITY CURVE CALCULATION:")
    print(f"Cash: ${current_cash:,.2f}")
    print(f"Positions Value: ${total_position_value:,.2f}")
    print(f"Total Portfolio Value: ${total_value:,.2f}")
    
    print()
    print("📈 PERFORMANCE:")
    print(f"Initial Value: ${initial_cash:,.2f}")
    print(f"Current Value: ${total_value:,.2f}")
    print(f"Total Return: ${total_value - initial_cash:,.2f} ({((total_value - initial_cash) / initial_cash * 100):+.2f}%)")
    
    if trade_history:
        print()
        print("📋 RECENT TRADES:")
        for trade in trade_history[-5:]:  # Show last 5 trades
            print(f"  {trade['date']}: {trade['action']} {trade['shares']} {trade['ticker']} @ ${trade['price']:.2f}")

if __name__ == "__main__":
    check_fixed_portfolio_status()
