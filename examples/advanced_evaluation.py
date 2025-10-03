"""
Advanced Evaluation Example for TradingAgents

This example demonstrates advanced evaluation features including:
- Configuration comparison
- Walk-forward analysis
- Learning evaluation
- Custom metrics
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import TradingEvaluator, BacktestConfig, BacktestEngine
from tradingagents.default_config import DEFAULT_CONFIG


def compare_configurations():
    """Compare different TradingAgents configurations."""
    
    print("Configuration Comparison Example")
    print("=" * 40)
    
    # Define different configurations to test
    configurations = {
        "Conservative": {
            "max_debate_rounds": 2,
            "max_risk_discuss_rounds": 2,
            "deep_think_llm": "gpt-4o-mini",
            "quick_think_llm": "gpt-4o-mini"
        },
        "Aggressive": {
            "max_debate_rounds": 1,
            "max_risk_discuss_rounds": 1,
            "deep_think_llm": "gpt-4o-mini",
            "quick_think_llm": "gpt-4o-mini"
        },
        "Balanced": {
            "max_debate_rounds": 1,
            "max_risk_discuss_rounds": 2,
            "deep_think_llm": "gpt-4o-mini",
            "quick_think_llm": "gpt-4o-mini"
        }
    }
    
    # Initialize evaluator
    evaluator = TradingEvaluator()
    
    # Test parameters
    tickers = ["AAPL", "AMZN","GOOGL"]
    start_date = datetime(2024, 3, 1)
    end_date = datetime(2024, 5, 31)
    
    print(f"Comparing {len(configurations)} configurations...")
    print(f"Tickers: {tickers}")
    print(f"Period: {start_date} to {end_date}")
    print()
    
    try:
        # Run comparison
        comparison_df = evaluator.compare_configurations(
            configs=configurations,
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            initial_cash=50000.0,
            analysts=["market", "news"],
            position_size_method="percent_portfolio",
            position_size_value=25.0  # 25% of portfolio per position
        )
        
        print("CONFIGURATION COMPARISON RESULTS:")
        print("-" * 50)
        print(comparison_df.to_string(index=False))
        
        # Find best configuration
        best_config = comparison_df.loc[comparison_df['Sharpe Ratio'].idxmax(), 'Configuration']
        print(f"\nBest Configuration (by Sharpe Ratio): {best_config}")
        
    except Exception as e:
        print(f"Error in configuration comparison: {e}")
        import traceback
        traceback.print_exc()


def evaluate_with_learning():
    """Evaluate strategy with learning enabled."""
    
    print("\n" + "=" * 50)
    print("Learning Evaluation Example")
    print("=" * 50)
    
    # Initialize evaluator
    evaluator = TradingEvaluator()
    
    # Test parameters
    tickers = ["AAPL"]
    start_date = datetime(2024, 1, 1)
    end_date = datetime(2024, 4, 30)
    
    print(f"Evaluating with learning enabled...")
    print(f"Ticker: {tickers[0]}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Learning frequency: Every 15 days")
    print()
    
    try:
        # Run evaluation with learning
        result = evaluator.evaluate_with_learning(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            learning_frequency=15,  # Update learning every 15 days
            initial_cash=100000.0,
            analysts=["market", "news"]
        )
        
        # Calculate realized profits from completed trades
        realized_profits = 0.0
        if not result.trades.empty:
            for ticker in result.trades['Ticker'].unique():
                ticker_trades = result.trades[result.trades['Ticker'] == ticker].sort_values('Timestamp')
                
                buy_price = None
                for _, trade in ticker_trades.iterrows():
                    if trade['Action'] == 'BUY':
                        buy_price = trade['Price']
                    elif trade['Action'] == 'SELL' and buy_price is not None:
                        trade_pnl = (trade['Price'] - buy_price) * trade['Shares']
                        realized_profits += trade_pnl
                        buy_price = None
        
        # Calculate total portfolio value including realized profits
        current_positions_value = sum(pos.market_value for pos in result.portfolio.positions.values())
        total_portfolio_value = result.portfolio.cash + current_positions_value + realized_profits
        
        print("LEARNING EVALUATION RESULTS:")
        print("-" * 40)
        print(f"Cash: ${result.portfolio.cash:,.2f}")
        print(f"Current Positions Value: ${current_positions_value:,.2f}")
        print(f"Realized Profits: ${realized_profits:,.2f}")
        print(f"Total Portfolio Value: ${total_portfolio_value:,.2f}")
        print(f"Total Return: {((total_portfolio_value / 100000.0) - 1) * 100:.2f}%")
        print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.3f}")
        print(f"Total Trades: {result.metrics.total_trades}")
        
        # Show learning progression (if available in signals_log)
        if result.signals_log:
            print(f"\nDecision History: {len(result.signals_log)} decisions recorded")
            
            # Count decisions by type
            decisions = [entry['decisions'] for entry in result.signals_log if 'decisions' in entry]
            if decisions:
                all_decisions = []
                for day_decisions in decisions:
                    all_decisions.extend(day_decisions.values())
                
                decision_counts = pd.Series(all_decisions).value_counts()
                print("Decision Distribution:")
                for decision, count in decision_counts.items():
                    print(f"  {decision}: {count}")
        
    except Exception as e:
        print(f"Error in learning evaluation: {e}")
        import traceback
        traceback.print_exc()


def run_walk_forward_analysis():
    """Run walk-forward analysis."""
    
    print("\n" + "=" * 50)
    print("Walk-Forward Analysis Example")
    print("=" * 50)
    
    # Initialize components
    evaluator = TradingEvaluator()
    backtest_engine = BacktestEngine(evaluator.data_manager)
    
    # Create TradingAgents instance
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    
    trading_config = DEFAULT_CONFIG.copy()
    trading_config.update({
        "deep_think_llm": "gpt-4o-mini",
        "quick_think_llm": "gpt-4o-mini",
        "max_debate_rounds": 1,
        "max_risk_discuss_rounds": 1
    })
    
    trading_graph = TradingAgentsGraph(
        selected_analysts=["market"],
        debug=False,
        config=trading_config
    )
    
    def trading_agent_func(ticker, trade_date):
        """Wrapper for TradingAgents."""
        try:
            final_state, decision = trading_graph.propagate(ticker, trade_date.strftime('%Y-%m-%d'))
            processed_decision = trading_graph.process_signal(decision)
            return processed_decision, final_state
        except:
            return "HOLD", {}
    
    # Configuration for walk-forward
    base_config = BacktestConfig(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 6, 30),
        initial_cash=100000.0,
        position_size_method="fixed_dollar",
        position_size_value=25000.0
    )
    
    tickers = ["AAPL"]
    
    print(f"Running walk-forward analysis...")
    print(f"Ticker: {tickers[0]}")
    print(f"Period: {base_config.start_date} to {base_config.end_date}")
    print(f"Train period: 60 days, Test period: 30 days")
    print()
    
    try:
        # Run walk-forward analysis
        results = backtest_engine.run_walk_forward_analysis(
            trading_agent_func=trading_agent_func,
            tickers=tickers,
            base_config=base_config,
            train_period_days=60,
            test_period_days=30
        )
        
        print("WALK-FORWARD ANALYSIS RESULTS:")
        print("-" * 40)
        print(f"Number of periods: {len(results)}")
        
        if results:
            # Aggregate results
            total_returns = [r.metrics.total_return_pct for r in results]
            sharpe_ratios = [r.metrics.sharpe_ratio for r in results]
            max_drawdowns = [r.metrics.max_drawdown for r in results]
            
            print(f"Average Return: {sum(total_returns)/len(total_returns):.2f}%")
            print(f"Average Sharpe: {sum(sharpe_ratios)/len(sharpe_ratios):.3f}")
            print(f"Average Max DD: {sum(max_drawdowns)/len(max_drawdowns):.2f}%")
            print(f"Win Rate: {sum(1 for r in total_returns if r > 0)/len(total_returns)*100:.1f}%")
            
            # Show period-by-period results
            print("\nPeriod-by-Period Results:")
            for i, result in enumerate(results):
                print(f"  Period {i+1}: {result.metrics.total_return_pct:6.2f}% return, "
                      f"{result.metrics.sharpe_ratio:5.2f} Sharpe, "
                      f"{result.metrics.max_drawdown:6.2f}% max DD")
        
    except Exception as e:
        print(f"Error in walk-forward analysis: {e}")
        import traceback
        traceback.print_exc()


def analyze_agent_performance():
    """Analyze individual agent performance (conceptual example)."""
    
    print("\n" + "=" * 50)
    print("Agent Performance Analysis Example")
    print("=" * 50)
    
    print("This example shows how you could analyze individual agent performance.")
    print("Note: This requires modifications to the TradingAgents framework to")
    print("capture individual agent predictions and track their accuracy.")
    print()
    
    # Example of what agent performance analysis could look like
    agent_performance = {
        "Market Analyst": {
            "predictions": 45,
            "correct": 28,
            "accuracy": 62.2,
            "avg_confidence": 0.73
        },
        "News Analyst": {
            "predictions": 45,
            "correct": 31,
            "accuracy": 68.9,
            "avg_confidence": 0.68
        },
        "Bull Researcher": {
            "arguments": 23,
            "successful": 15,
            "success_rate": 65.2,
            "avg_strength": 0.71
        },
        "Bear Researcher": {
            "arguments": 22,
            "successful": 12,
            "success_rate": 54.5,
            "avg_strength": 0.66
        }
    }
    
    print("HYPOTHETICAL AGENT PERFORMANCE:")
    print("-" * 40)
    
    for agent, metrics in agent_performance.items():
        print(f"\n{agent}:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                print(f"  {metric}: {value:.1f}")
            else:
                print(f"  {metric}: {value}")
    
    print("\nTo implement this analysis, you would need to:")
    print("1. Modify agents to record individual predictions")
    print("2. Track prediction accuracy over time")
    print("3. Analyze which agents contribute most to performance")
    print("4. Identify patterns in agent behavior")


def main():
    """Run all advanced evaluation examples."""
    
    print("TradingAgents Advanced Evaluation Examples")
    print("=" * 60)
    
    # Run different evaluation examples
    try:
        compare_configurations()
        evaluate_with_learning()
        run_walk_forward_analysis()
        analyze_agent_performance()
        
        print("\n" + "=" * 60)
        print("All advanced evaluation examples completed!")
        
    except KeyboardInterrupt:
        print("\nEvaluation interrupted by user.")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
