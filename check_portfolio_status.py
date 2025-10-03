"""
Quick Portfolio Status Checker
Run this while basic_evaluation.py is running to see current status
"""

import json
import os
from datetime import datetime
from pathlib import Path
import pandas as pd

def check_current_status():
    """Check current portfolio status from log files."""
    
    print("💰 TradingAgents Portfolio Status Check")
    print("=" * 50)
    print(f"🕐 Checked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check eval_results directory
    results_dir = Path("eval_results")
    
    if not results_dir.exists():
        print("❌ No evaluation results found yet...")
        return
    
    # Find all tickers
    tickers = [d.name for d in results_dir.iterdir() if d.is_dir()]
    print(f"📊 Tracking tickers: {tickers}")
    print()
    
    # Simulate portfolio tracking
    initial_cash = 100000.0
    current_cash = initial_cash
    positions = {}  # {ticker: shares}
    trades = []
    
    # Process all log files to reconstruct portfolio
    for ticker in tickers:
        ticker_dir = results_dir / ticker / "TradingAgentsStrategy_logs"
        
        if not ticker_dir.exists():
            continue
            
        # Get all log files for this ticker
        log_files = list(ticker_dir.glob("full_states_log_*.json"))
        log_files.sort(key=lambda x: x.stem.split('_')[-1])  # Sort by date
        
        print(f"📈 {ticker} Analysis:")
        print("-" * 30)
        
        ticker_trades = []
        
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)
                
                for date, state in data.items():
                    decision = state.get('final_trade_decision', '')
                    
                    # Extract decision
                    if '**Sell**' in decision or 'SELL' in decision:
                        action = 'SELL'
                    elif '**Buy**' in decision or 'BUY' in decision:
                        action = 'BUY'
                    else:
                        action = 'HOLD'
                    
                    # For simulation, assume $20k position size
                    position_size = 20000.0
                    
                    # Simulate trade execution (simplified)
                    if action == 'BUY' and ticker not in positions:
                        # Assume we can get a rough price estimate
                        shares = position_size / 200  # Rough estimate
                        positions[ticker] = shares
                        current_cash -= position_size
                        ticker_trades.append({
                            'Date': date,
                            'Action': action,
                            'Shares': shares,
                            'Value': position_size
                        })
                        print(f"  {date}: {action} {shares:.1f} shares (~${position_size:,.0f})")
                        
                    elif action == 'SELL' and ticker in positions:
                        shares = positions[ticker]
                        current_cash += position_size
                        del positions[ticker]
                        ticker_trades.append({
                            'Date': date,
                            'Action': action,
                            'Shares': shares,
                            'Value': position_size
                        })
                        print(f"  {date}: {action} {shares:.1f} shares (~${position_size:,.0f})")
                        
                    elif action == 'HOLD':
                        print(f"  {date}: {action}")
                        
            except Exception as e:
                print(f"  Error reading {log_file}: {e}")
                continue
        
        # Show ticker summary
        if ticker_trades:
            buy_trades = [t for t in ticker_trades if t['Action'] == 'BUY']
            sell_trades = [t for t in ticker_trades if t['Action'] == 'SELL']
            
            print(f"  📊 Trades: {len(ticker_trades)} total ({len(buy_trades)} buys, {len(sell_trades)} sells)")
            
            if ticker in positions:
                print(f"  💼 Current Position: {positions[ticker]:.1f} shares")
            else:
                print(f"  💼 Current Position: None")
        else:
            print(f"  📊 No trades yet")
        
        print()
    
    # Overall portfolio summary
    print("💼 PORTFOLIO SUMMARY:")
    print("-" * 30)
    print(f"💰 Initial Cash: ${initial_cash:,.2f}")
    print(f"💰 Current Cash: ${current_cash:,.2f}")
    print(f"📈 Cash Used: ${initial_cash - current_cash:,.2f}")
    print(f"💼 Active Positions: {len(positions)}")
    
    if positions:
        print("\n📊 Current Positions:")
        for ticker, shares in positions.items():
            print(f"  {ticker}: {shares:.1f} shares")
    
    # Calculate realized profits from completed trades
    realized_profits = 0.0
    if trade_history:
        # Group trades by ticker to calculate P&L
        for ticker in set(t['Ticker'] for t in trade_history):
            ticker_trades = [t for t in trade_history if t['Ticker'] == ticker]
            ticker_trades.sort(key=lambda x: x['Date'])
            
            buy_price = None
            for trade in ticker_trades:
                if trade['Action'] == 'BUY':
                    buy_price = trade['Price']
                elif trade['Action'] == 'SELL' and buy_price is not None:
                    # Calculate profit/loss from this trade
                    trade_pnl = (trade['Price'] - buy_price) * trade['Shares']
                    realized_profits += trade_pnl
                    buy_price = None  # Reset for next trade pair
    
    # Calculate proper equity curve = cash + Σ(positions × last_price) + realized_profits
    portfolio_value = current_cash + realized_profits
    
    # Add value of current positions (simplified - using rough price estimates)
    position_values = {}
    for ticker, shares in positions.items():
        # Rough price estimates (you'd get real prices in practice)
        if ticker == "AAPL":
            estimated_price = 200.0  # Rough estimate
        elif ticker == "MSFT":
            estimated_price = 400.0  # Rough estimate
        elif ticker == "GOOGL":
            estimated_price = 150.0  # Rough estimate
        else:
            estimated_price = 200.0  # Default estimate
        
        position_value = shares * estimated_price
        position_values[ticker] = position_value
        portfolio_value += position_value
    
    print(f"\n💎 PORTFOLIO VALUE CALCULATION:")
    print(f"💰 Cash: ${current_cash:,.2f}")
    print(f"💵 Realized Profits: ${realized_profits:,.2f}")
    
    if position_values:
        print(f"📊 Position Values:")
        for ticker, value in position_values.items():
            shares = positions[ticker]
            estimated_price = value / shares
            print(f"  {ticker}: {shares:.1f} shares × ${estimated_price:.2f} = ${value:,.2f}")
    
    print(f"\n💎 Total Portfolio Value: ${portfolio_value:,.2f}")
    print(f"📈 Total Return: ${portfolio_value - initial_cash:,.2f} ({(portfolio_value/initial_cash - 1)*100:.2f}%)")
    
    # Show recent activity
    print(f"\n📅 Recent Activity:")
    all_trades = []
    for ticker in tickers:
        ticker_dir = results_dir / ticker / "TradingAgentsStrategy_logs"
        if ticker_dir.exists():
            log_files = list(ticker_dir.glob("full_states_log_*.json"))
            log_files.sort(key=lambda x: x.stem.split('_')[-1], reverse=True)  # Most recent first
            
            if log_files:
                latest_file = log_files[0]
                try:
                    with open(latest_file, 'r') as f:
                        data = json.load(f)
                    
                    for date, state in data.items():
                        decision = state.get('final_trade_decision', '')
                        if '**Sell**' in decision or 'SELL' in decision:
                            action = 'SELL'
                        elif '**Buy**' in decision or 'BUY' in decision:
                            action = 'BUY'
                        else:
                            action = 'HOLD'
                        
                        all_trades.append({
                            'Date': date,
                            'Ticker': ticker,
                            'Action': action
                        })
                except:
                    pass
    
    # Sort by date and show recent trades
    all_trades.sort(key=lambda x: x['Date'], reverse=True)
    recent_trades = all_trades[:10]  # Last 10 decisions
    
    for trade in recent_trades:
        print(f"  {trade['Date']}: {trade['Ticker']} - {trade['Action']}")

def main():
    """Main function."""
    try:
        check_current_status()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
