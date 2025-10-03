#!/usr/bin/env python3
"""
Diagnose Trading Issues

This script analyzes the evaluation logs to identify why trades aren't being executed.
"""

import json
import re
from pathlib import Path
from collections import Counter

def analyze_decision_patterns():
    """Analyze decision patterns in evaluation logs."""
    
    print("🔍 TradingAgents Decision Analysis")
    print("=" * 50)
    
    # Check eval_results directory
    results_dir = Path("eval_results")
    
    if not results_dir.exists():
        print("❌ No evaluation results found yet...")
        return
    
    # Find all tickers
    tickers = [d.name for d in results_dir.iterdir() if d.is_dir()]
    print(f"📊 Found tickers: {tickers}")
    print()
    
    all_decisions = []
    decision_patterns = Counter()
    
    # Process all log files
    for ticker in tickers:
        ticker_dir = results_dir / ticker / "TradingAgentsStrategy_logs"
        
        if not ticker_dir.exists():
            continue
            
        log_files = list(ticker_dir.glob("full_states_log_*.json"))
        log_files.sort(key=lambda x: x.stem.split('_')[-1])
        
        print(f"📈 Analyzing {ticker}...")
        
        for log_file in log_files:
            try:
                with open(log_file, 'r') as f:
                    data = json.load(f)
                
                for date, state in data.items():
                    decision_text = state.get('final_trade_decision', '')
                    company = state.get('company_of_interest', 'Unknown')
                    
                    if decision_text:
                        # Extract decision using various patterns
                        decision = extract_decision_advanced(decision_text)
                        all_decisions.append({
                            'ticker': ticker,
                            'company': company,
                            'date': date,
                            'decision': decision,
                            'raw_text': decision_text[:200] + "..." if len(decision_text) > 200 else decision_text
                        })
                        
                        decision_patterns[decision] += 1
                        
                        print(f"  📅 {date} ({company}): {decision}")
                
            except Exception as e:
                print(f"    ❌ Error processing {log_file}: {e}")
    
    print()
    print("📊 DECISION SUMMARY:")
    print("-" * 30)
    for decision, count in decision_patterns.most_common():
        print(f"{decision}: {count} times")
    
    print()
    print("🔍 DETAILED ANALYSIS:")
    print("-" * 30)
    
    # Analyze by ticker vs company
    ticker_company_mismatch = []
    for decision in all_decisions:
        if decision['ticker'] != decision['company']:
            ticker_company_mismatch.append(decision)
    
    if ticker_company_mismatch:
        print(f"⚠️  Ticker/Company Mismatch: {len(ticker_company_mismatch)} cases")
        for mismatch in ticker_company_mismatch[:3]:  # Show first 3
            print(f"  {mismatch['ticker']} folder but analyzing {mismatch['company']}")
    else:
        print("✅ No ticker/company mismatches found")
    
    # Analyze decision quality
    unclear_decisions = []
    for decision in all_decisions:
        if decision['decision'] == 'UNCLEAR':
            unclear_decisions.append(decision)
    
    if unclear_decisions:
        print(f"⚠️  Unclear decisions: {len(unclear_decisions)} cases")
        for unclear in unclear_decisions[:3]:  # Show first 3
            print(f"  {unclear['date']}: {unclear['raw_text']}")
    else:
        print("✅ All decisions are clear")
    
    # Show sample decisions
    print()
    print("📋 SAMPLE DECISIONS:")
    print("-" * 30)
    for i, decision in enumerate(all_decisions[:5]):  # Show first 5
        print(f"{i+1}. {decision['date']} ({decision['company']}): {decision['decision']}")
        print(f"   Text: {decision['raw_text']}")
        print()

def extract_decision_advanced(decision_text):
    """Advanced decision extraction with more patterns."""
    if not decision_text:
        return 'EMPTY'
    
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
        r'DECISION:\s*SELL',
        r'RECOMMENDATION:\s*\*\*.*SELL.*\*\*',
        r'FINAL\s+TRANSACTION\s+PROPOSAL:\s*\*\*.*SELL.*\*\*'
    ]
    
    # Look for explicit BUY indicators
    buy_patterns = [
        r'\*\*BUY\*\*',
        r'BUY\s+\w+',  # BUY AAPL, BUY GOOGL, etc.
        r'RECOMMENDATION:\s*\*\*BUY\*\*',
        r'FINAL\s+TRANSACTION\s+PROPOSAL:\s*\*\*BUY\*\*',
        r'RECOMMENDATION:\s*BUY',
        r'ACTION:\s*BUY',
        r'DECISION:\s*BUY',
        r'RECOMMENDATION:\s*\*\*.*BUY.*\*\*',
        r'FINAL\s+TRANSACTION\s+PROPOSAL:\s*\*\*.*BUY.*\*\*'
    ]
    
    # Look for HOLD indicators
    hold_patterns = [
        r'\*\*HOLD\*\*',
        r'HOLD\s+\w+',  # HOLD AAPL, HOLD GOOGL, etc.
        r'RECOMMENDATION:\s*\*\*HOLD\*\*',
        r'RECOMMENDATION:\s*HOLD',
        r'ACTION:\s*HOLD',
        r'DECISION:\s*HOLD',
        r'RECOMMENDATION:\s*\*\*.*HOLD.*\*\*'
    ]
    
    # Check for SELL first (most specific)
    for pattern in sell_patterns:
        if re.search(pattern, text_upper):
            return 'SELL'
    
    # Check for BUY
    for pattern in buy_patterns:
        if re.search(pattern, text_upper):
            return 'BUY'
    
    # Check for HOLD
    for pattern in hold_patterns:
        if re.search(pattern, text_upper):
            return 'HOLD'
    
    # If no clear pattern found, return UNCLEAR
    return 'UNCLEAR'

if __name__ == "__main__":
    analyze_decision_patterns()
