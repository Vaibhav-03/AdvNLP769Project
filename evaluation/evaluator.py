"""
Main Evaluator for TradingAgents Framework

Orchestrates the complete evaluation process including backtesting, 
performance analysis, and agent learning.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Tuple
from pathlib import Path
import json

# Import TradingAgents components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG

from .backtesting import BacktestEngine, BacktestConfig, BacktestResult
from .portfolio import Portfolio, PortfolioManager
from .data_manager import DataManager
from .metrics import MetricsCalculator, PerformanceMetrics
from .visualization import PerformanceVisualizer


class TradingEvaluator:
    """Main evaluator class that orchestrates the evaluation process."""
    
    def __init__(self, 
                 trading_config: Optional[Dict] = None,
                 cache_dir: str = "./evaluation/cache",
                 results_dir: str = "./evaluation/results"):
        """
        Initialize the evaluator.
        
        Args:
            trading_config: Configuration for TradingAgents
            cache_dir: Directory for data caching
            results_dir: Directory for saving results
        """
        # Initialize TradingAgents config
        self.trading_config = trading_config or DEFAULT_CONFIG.copy()
        
        # Initialize evaluation components
        self.data_manager = DataManager(cache_dir)
        self.backtest_engine = BacktestEngine(self.data_manager)
        self.portfolio_manager = PortfolioManager()
        self.metrics_calculator = MetricsCalculator()
        self.visualizer = PerformanceVisualizer()
        
        # Create results directory
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Track evaluation runs
        self.evaluation_history = []
    
    def evaluate_strategy(self,
                         tickers: List[str],
                         start_date: datetime,
                         end_date: datetime,
                         initial_cash: float = 100000.0,
                         analysts: List[str] = ["market", "social", "news", "fundamentals"],
                         **kwargs) -> BacktestResult:
        """
        Evaluate TradingAgents strategy on historical data.
        
        Args:
            tickers: List of stock tickers to trade
            start_date: Start date for evaluation
            end_date: End date for evaluation
            initial_cash: Initial portfolio cash
            analysts: List of analysts to use
            **kwargs: Additional configuration parameters
            
        Returns:
            BacktestResult object with complete evaluation results
        """
        print(f"Evaluating TradingAgents strategy")
        print(f"Period: {start_date} to {end_date}")
        print(f"Tickers: {tickers}")
        print(f"Analysts: {analysts}")
        
        # Create backtest configuration
        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_cash=initial_cash,
            **kwargs
        )
        
        # Create TradingAgents instance
        trading_graph = TradingAgentsGraph(
            selected_analysts=analysts,
            debug=False,  # Set to False for faster execution
            config=self.trading_config
        )
        
        # Define trading function
        def trading_agent_func(ticker: str, trade_date: datetime) -> Tuple[str, Dict]:
            """Wrapper function for TradingAgents."""
            try:
                final_state, decision = trading_graph.propagate(ticker, trade_date.strftime('%Y-%m-%d'))
                
                # Extract decision using signal processor
                processed_decision = trading_graph.process_signal(decision)
                
                return processed_decision, final_state
                
            except Exception as e:
                print(f"Error in trading agent for {ticker} on {trade_date}: {e}")
                return "HOLD", {}
        
        # Run backtest
        result = self.backtest_engine.run_backtest(
            trading_agent_func=trading_agent_func,
            tickers=tickers,
            config=config
        )
        
        # Store evaluation history
        self.evaluation_history.append({
            'timestamp': datetime.now(),
            'tickers': tickers,
            'start_date': start_date,
            'end_date': end_date,
            'analysts': analysts,
            'final_value': result.portfolio.total_value,
            'total_return_pct': result.metrics.total_return_pct,
            'sharpe_ratio': result.metrics.sharpe_ratio,
            'max_drawdown': result.metrics.max_drawdown
        })
        
        return result
    
    def evaluate_with_learning(self,
                              tickers: List[str],
                              start_date: datetime,
                              end_date: datetime,
                              learning_frequency: int = 30,  # Days between learning updates
                              **kwargs) -> BacktestResult:
        """
        Evaluate strategy with periodic learning updates.
        
        Args:
            tickers: List of stock tickers
            start_date: Start date
            end_date: End date
            learning_frequency: Days between learning updates
            **kwargs: Additional configuration
            
        Returns:
            BacktestResult with learning incorporated
        """
        print(f"Evaluating with learning (update every {learning_frequency} days)")
        
        # Create TradingAgents instance
        trading_graph = TradingAgentsGraph(
            selected_analysts=kwargs.get('analysts', ["market", "social", "news", "fundamentals"]),
            debug=False,
            config=self.trading_config
        )
        
        # Create portfolio for tracking performance
        portfolio = Portfolio(initial_cash=kwargs.get('initial_cash', 100000.0))
        
        # Get trading days
        trading_days = self.data_manager.get_trading_days(start_date, end_date)
        
        # Track decisions and performance for learning
        decision_history = []
        performance_history = []
        
        last_learning_update = start_date
        
        for i, current_date in enumerate(trading_days):
            print(f"Processing {current_date.strftime('%Y-%m-%d')} ({i+1}/{len(trading_days)})")
            
            # Get decisions for all tickers
            daily_decisions = {}
            daily_states = {}
            
            for ticker in tickers:
                try:
                    final_state, decision = trading_graph.propagate(ticker, current_date.strftime('%Y-%m-%d'))
                    processed_decision = trading_graph.process_signal(decision)
                    
                    daily_decisions[ticker] = processed_decision
                    daily_states[ticker] = final_state
                    
                except Exception as e:
                    print(f"Error getting decision for {ticker}: {e}")
                    daily_decisions[ticker] = "HOLD"
                    daily_states[ticker] = {}
            
            # Execute trades and calculate performance
            day_performance = self._execute_and_evaluate_day(
                portfolio, daily_decisions, current_date, tickers
            )
            
            # Store for learning
            decision_history.append({
                'date': current_date,
                'decisions': daily_decisions.copy(),
                'states': daily_states.copy()
            })
            performance_history.append(day_performance)
            
            # Check if it's time for learning update
            days_since_update = (current_date - last_learning_update).days
            if days_since_update >= learning_frequency and len(performance_history) >= learning_frequency:
                print(f"  Updating agent learning...")
                
                # Calculate recent performance for learning
                recent_performance = performance_history[-learning_frequency:]
                avg_return = np.mean([p['total_return'] for p in recent_performance])
                
                # Update agent memories with recent performance
                try:
                    trading_graph.reflect_and_remember(avg_return)
                    print(f"  Learning update completed (avg return: {avg_return:.4f})")
                except Exception as e:
                    print(f"  Learning update failed: {e}")
                
                last_learning_update = current_date
        
        # Create final result
        daily_values_df = portfolio.get_daily_values()
        trades_df = portfolio.get_trade_history()
        
        # Calculate metrics
        if not daily_values_df.empty:
            portfolio_values = daily_values_df.set_index('Date')['Portfolio Value']
            benchmark_data = self.data_manager.get_stock_data('SPY', start_date, end_date)
            benchmark_returns = benchmark_data['Close'].pct_change().dropna()
            
            metrics = self.metrics_calculator.calculate_metrics(
                portfolio_values, trades_df, benchmark_returns
            )
        else:
            metrics = PerformanceMetrics(
                total_return=0, total_return_pct=0, annualized_return=0,
                volatility=0, sharpe_ratio=0, sortino_ratio=0,
                max_drawdown=0, max_drawdown_duration=0,
                total_trades=0, win_rate=0, avg_win=0, avg_loss=0, profit_factor=0,
                calmar_ratio=0, information_ratio=0
            )
        
        # Create config for result
        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_cash=kwargs.get('initial_cash', 100000.0)
        )
        
        result = BacktestResult(
            config=config,
            portfolio=portfolio,
            metrics=metrics,
            daily_values=daily_values_df,
            trades=trades_df,
            benchmark_data=benchmark_data,
            signals_log=decision_history
        )
        
        return result
    
    def _execute_and_evaluate_day(self, 
                                 portfolio: Portfolio,
                                 decisions: Dict[str, str],
                                 current_date: datetime,
                                 tickers: List[str]) -> Dict:
        """Execute trades for a day and evaluate performance."""
        
        # Get current prices
        current_prices = {}
        for ticker in tickers:
            try:
                price = self.data_manager.get_price_at_date(ticker, current_date)
                current_prices[ticker] = price
            except:
                continue
        
        # Execute trades
        for ticker, decision in decisions.items():
            if decision == "HOLD" or ticker not in current_prices:
                continue
            
            price = current_prices[ticker]
            
            # Simple position sizing: $10,000 per trade
            if decision == "BUY":
                shares = 10000 / price
                portfolio.execute_trade(ticker, "BUY", shares, price, current_date)
            elif decision == "SELL" and ticker in portfolio.positions:
                shares = portfolio.positions[ticker].shares
                portfolio.execute_trade(ticker, "SELL", shares, price, current_date)
        
        # Update portfolio prices
        portfolio.update_prices(current_prices, current_date)
        
        # Calculate day's performance
        return {
            'date': current_date,
            'portfolio_value': portfolio.total_value,
            'total_return': portfolio.total_return,
            'decisions_count': len([d for d in decisions.values() if d != "HOLD"])
        }
    
    def compare_configurations(self,
                              configs: Dict[str, Dict],
                              tickers: List[str],
                              start_date: datetime,
                              end_date: datetime,
                              **kwargs) -> pd.DataFrame:
        """
        Compare different TradingAgents configurations.
        
        Args:
            configs: Dictionary mapping config names to config dictionaries
            tickers: List of tickers to test
            start_date: Start date
            end_date: End date
            **kwargs: Additional parameters
            
        Returns:
            DataFrame comparing all configurations
        """
        print(f"Comparing {len(configs)} configurations")
        
        results = {}
        
        for config_name, config_dict in configs.items():
            print(f"\nEvaluating configuration: {config_name}")
            
            # Update trading config
            temp_config = self.trading_config.copy()
            temp_config.update(config_dict)
            
            # Save original config
            original_config = self.trading_config
            self.trading_config = temp_config
            
            try:
                # Run evaluation
                result = self.evaluate_strategy(
                    tickers=tickers,
                    start_date=start_date,
                    end_date=end_date,
                    **kwargs
                )
                
                results[config_name] = result
                
            except Exception as e:
                print(f"Error evaluating {config_name}: {e}")
                continue
            finally:
                # Restore original config
                self.trading_config = original_config
        
        # Create comparison DataFrame
        comparison_data = []
        for name, result in results.items():
            comparison_data.append({
                'Configuration': name,
                'Final Value': result.portfolio.total_value,
                'Total Return %': result.metrics.total_return_pct,
                'Sharpe Ratio': result.metrics.sharpe_ratio,
                'Max Drawdown %': result.metrics.max_drawdown,
                'Total Trades': result.metrics.total_trades,
                'Win Rate %': result.metrics.win_rate,
                'Volatility %': result.metrics.volatility
            })
        
        return pd.DataFrame(comparison_data)
    
    def generate_report(self, 
                       result: BacktestResult,
                       save_plots: bool = True) -> str:
        """
        Generate comprehensive evaluation report.
        
        Args:
            result: BacktestResult to generate report for
            save_plots: Whether to save visualization plots
            
        Returns:
            Report as string
        """
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("TRADINGAGENTS EVALUATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        # Configuration section
        report_lines.append("CONFIGURATION:")
        report_lines.append(f"  Period: {result.config.start_date} to {result.config.end_date}")
        report_lines.append(f"  Initial Cash: ${result.config.initial_cash:,.2f}")
        report_lines.append(f"  Commission: ${result.config.commission_per_trade:.2f}")
        report_lines.append(f"  Position Sizing: {result.config.position_size_method}")
        report_lines.append("")
        
        # Performance section
        report_lines.append("PERFORMANCE METRICS:")
        metrics_dict = result.metrics.to_dict()
        for key, value in metrics_dict.items():
            report_lines.append(f"  {key}: {value}")
        report_lines.append("")
        
        # Trading activity
        report_lines.append("TRADING ACTIVITY:")
        report_lines.append(f"  Total Trades: {len(result.trades)}")
        if not result.trades.empty:
            buy_trades = len(result.trades[result.trades['Action'] == 'BUY'])
            sell_trades = len(result.trades[result.trades['Action'] == 'SELL'])
            report_lines.append(f"  Buy Trades: {buy_trades}")
            report_lines.append(f"  Sell Trades: {sell_trades}")
        report_lines.append("")
        
        # Portfolio summary
        report_lines.append("FINAL PORTFOLIO:")
        report_lines.append(f"  Cash: ${result.portfolio.cash:,.2f}")
        report_lines.append(f"  Total Value: ${result.portfolio.total_value:,.2f}")
        report_lines.append(f"  Positions: {len(result.portfolio.positions)}")
        report_lines.append("")
        
        # Generate visualizations if requested
        if save_plots and not result.daily_values.empty:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_dir = self.results_dir / f"plots_{timestamp}"
            plot_dir.mkdir(exist_ok=True)
            
            try:
                # Portfolio value chart
                self.visualizer.plot_portfolio_value(
                    result.daily_values,
                    result.benchmark_data,
                    save_path=plot_dir / "portfolio_value.png"
                )
                
                # Performance metrics chart
                self.visualizer.plot_performance_metrics(
                    result.metrics,
                    save_path=plot_dir / "performance_metrics.png"
                )
                
                report_lines.append(f"Plots saved to: {plot_dir}")
                
            except Exception as e:
                report_lines.append(f"Error generating plots: {e}")
        
        report_lines.append("=" * 60)
        
        # Save report
        report_text = "\n".join(report_lines)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = self.results_dir / f"evaluation_report_{timestamp}.txt"
        
        with open(report_path, 'w') as f:
            f.write(report_text)
        
        print(f"Report saved to: {report_path}")
        
        return report_text
    
    def get_evaluation_summary(self) -> pd.DataFrame:
        """Get summary of all evaluation runs."""
        if not self.evaluation_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.evaluation_history)
