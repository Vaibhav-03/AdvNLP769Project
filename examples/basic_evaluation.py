"""
Basic Evaluation Example for TradingAgents

This example demonstrates how to run a basic evaluation of the TradingAgents system.
"""

import sys
import os
from datetime import datetime, timedelta

# Add the parent directory to the path to import evaluation modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import TradingEvaluator
from tradingagents.default_config import DEFAULT_CONFIG


def main():
    """Run basic evaluation example."""
    
    print("TradingAgents Basic Evaluation Example")
    print("=" * 50)
    
    # Configuration
    tickers = ["AAPL"]
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 3, 29)
    initial_cash = 100000.0
    
    # Create custom trading configuration
    trading_config = DEFAULT_CONFIG.copy()
    trading_config.update({
        "deep_think_llm": "gpt-4o-mini",  # Use cheaper model for testing
        "quick_think_llm": "gpt-4o-mini",
        "max_debate_rounds": 1,  # Reduce for faster execution
        "max_risk_discuss_rounds": 1,
        "online_tools": True  # Use real-time data
    })
    
    # Initialize evaluator
    evaluator = TradingEvaluator(
        trading_config=trading_config,
        cache_dir="./evaluation/cache",
        results_dir="./evaluation/results"
    )
    
    print(f"Evaluating tickers: {tickers}")
    print(f"Period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    print(f"Initial cash: ${initial_cash:,.2f}")
    print()
    
    try:
        # Run evaluation
        result = evaluator.evaluate_strategy(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            initial_cash=initial_cash,
            analysts=["market", "news","social","fundamentals"],  # Use subset of analysts for faster execution
            position_size_method="fixed_dollar",
            position_size_value=20000.0,  # $20k per position
            commission_per_trade=0.0
        )
        
        # Print results
        print("EVALUATION RESULTS:")
        print("-" * 30)
        print(f"Final Portfolio Value: ${result.portfolio.total_value:,.2f}")
        print(f"Total Return: ${result.metrics.total_return:,.2f}")
        print(f"Total Return %: {result.metrics.total_return_pct:.2f}%")
        print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.3f}")
        print(f"Maximum Drawdown: {result.metrics.max_drawdown:.2f}%")
        print(f"Win Rate: {result.metrics.win_rate:.2f}%")
        print(f"Total Trades: {result.metrics.total_trades}")
        print()
        
        # Generate comprehensive report
        print("Generating detailed report...")
        report = evaluator.generate_report(result, save_plots=True)
        print("Report generated successfully!")
        
        # Show portfolio positions
        if result.portfolio.positions:
            print("\nFINAL POSITIONS:")
            positions_df = result.portfolio.get_position_summary()
            print(positions_df.to_string(index=False))
        
        # Show recent trades
        if not result.trades.empty:
            print("\nRECENT TRADES (Last 10):")
            recent_trades = result.trades.tail(10)
            print(recent_trades[['Timestamp', 'Ticker', 'Action', 'Shares', 'Price']].to_string(index=False))
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
