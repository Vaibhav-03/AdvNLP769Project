"""
Real-time evaluation monitoring script
Run this in a separate terminal while evaluation is running
"""

import json
import os
from datetime import datetime
import pandas as pd
from pathlib import Path

def monitor_evaluation_results():
    """Monitor evaluation results in real-time."""
    
    results_dir = Path("eval_results")
    
    if not results_dir.exists():
        print("No evaluation results found yet...")
        return
    
    print("🔍 TradingAgents Evaluation Monitor")
    print("=" * 50)
    
    # Find all ticker directories
    tickers = [d.name for d in results_dir.iterdir() if d.is_dir()]
    
    if not tickers:
        print("No ticker data found yet...")
        return
    
    print(f"Monitoring tickers: {tickers}")
    print()
    
    for ticker in tickers:
        ticker_dir = results_dir / ticker / "TradingAgentsStrategy_logs"
        
        if not ticker_dir.exists():
            continue
            
        # Get all log files
        log_files = list(ticker_dir.glob("full_states_log_*.json"))
        
        if not log_files:
            continue
            
        # Sort by date
        log_files.sort(key=lambda x: x.stem.split('_')[-1])
        
        print(f"📊 {ticker} Results:")
        print("-" * 30)
        
        # Process each log file
        decisions = []
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)
                
                for date, state in data.items():
                    decision = state.get('final_trade_decision', 'N/A')
                    
                    # Extract decision from the text
                    if '**Sell**' in decision or 'SELL' in decision:
                        action = 'SELL'
                    elif '**Buy**' in decision or 'BUY' in decision:
                        action = 'BUY'
                    else:
                        action = 'HOLD'
                    
                    decisions.append({
                        'Date': date,
                        'Action': action,
                        'Decision_Text': decision[:100] + '...' if len(decision) > 100 else decision
                    })
                    
            except Exception as e:
                print(f"Error reading {log_file}: {e}")
                continue
        
        if decisions:
            df = pd.DataFrame(decisions)
            
            # Show recent decisions
            print(f"Recent decisions ({len(decisions)} total):")
            recent = df.tail(5)
            for _, row in recent.iterrows():
                print(f"  {row['Date']}: {row['Action']}")
            
            # Show decision summary
            action_counts = df['Action'].value_counts()
            print(f"\nDecision Summary:")
            for action, count in action_counts.items():
                print(f"  {action}: {count} ({count/len(decisions)*100:.1f}%)")
            
            print()
        else:
            print("No decisions found yet...")
            print()

def check_portfolio_value():
    """Check current portfolio value if available."""
    
    # Look for any portfolio tracking files
    portfolio_files = list(Path(".").glob("**/portfolio_*.json"))
    
    if portfolio_files:
        print("💰 Portfolio Status:")
        print("-" * 20)
        
        for file in portfolio_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                
                if 'total_value' in data:
                    print(f"Total Value: ${data['total_value']:,.2f}")
                if 'cash' in data:
                    print(f"Cash: ${data['cash']:,.2f}")
                if 'positions' in data:
                    print(f"Positions: {len(data['positions'])}")
                    
            except Exception as e:
                print(f"Error reading {file}: {e}")
    else:
        print("No portfolio tracking files found yet...")

def main():
    """Main monitoring loop."""
    import time
    
    print("Starting evaluation monitor...")
    print("Press Ctrl+C to stop")
    print()
    
    try:
        while True:
            os.system('clear' if os.name == 'posix' else 'cls')  # Clear screen
            
            print(f"🕐 Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            
            monitor_evaluation_results()
            check_portfolio_value()
            
            print("\nRefreshing in 10 seconds...")
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped.")

if __name__ == "__main__":
    main()
