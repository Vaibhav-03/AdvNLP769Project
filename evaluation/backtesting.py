"""
Backtesting Engine for TradingAgents Evaluation Framework

Provides comprehensive backtesting capabilities for the TradingAgents system.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from pathlib import Path
import json

from .portfolio import Portfolio, TradeAction
from .data_manager import DataManager
from .metrics import MetricsCalculator, PerformanceMetrics


@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    
    # Time period
    start_date: datetime
    end_date: datetime
    
    # Portfolio settings
    initial_cash: float = 100000.0
    commission_per_trade: float = 0.0
    
    # Position sizing
    position_size_method: str = "fixed_dollar"  # "fixed_dollar", "fixed_shares", "percent_portfolio"
    position_size_value: float = 10000.0  # Dollar amount, share count, or percentage
    
    # Risk management
    max_position_size: float = 0.2  # Maximum position size as % of portfolio
    stop_loss_pct: Optional[float] = None  # Stop loss percentage
    take_profit_pct: Optional[float] = None  # Take profit percentage
    
    # Execution settings
    execution_delay: int = 0  # Days delay between signal and execution
    slippage_pct: float = 0.0  # Slippage percentage
    
    # Rebalancing
    rebalance_frequency: str = "daily"  # "daily", "weekly", "monthly"
    
    # Benchmark
    benchmark_ticker: str = "SPY"
    
    # Other settings
    lookback_days: int = 1  # Days to look back for performance evaluation
    save_results: bool = True
    results_dir: str = "./evaluation/results"


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    
    config: BacktestConfig
    portfolio: Portfolio
    metrics: PerformanceMetrics
    daily_values: pd.DataFrame
    trades: pd.DataFrame
    benchmark_data: pd.DataFrame
    signals_log: List[Dict] = field(default_factory=list)
    
    def save_to_file(self, filepath: str):
        """Save results to file."""
        results_dict = {
            'config': {
                'start_date': self.config.start_date.isoformat(),
                'end_date': self.config.end_date.isoformat(),
                'initial_cash': self.config.initial_cash,
                'commission_per_trade': self.config.commission_per_trade,
                'position_size_method': self.config.position_size_method,
                'position_size_value': self.config.position_size_value,
                'benchmark_ticker': self.config.benchmark_ticker
            },
            'metrics': self.metrics.to_dict(),
            'final_portfolio_value': self.portfolio.total_value,
            'total_trades': len(self.trades),
            'signals_count': len(self.signals_log)
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Save detailed data as CSV
        base_path = Path(filepath).parent / Path(filepath).stem
        self.daily_values.to_csv(f"{base_path}_daily_values.csv")
        self.trades.to_csv(f"{base_path}_trades.csv")
        self.benchmark_data.to_csv(f"{base_path}_benchmark.csv")


class BacktestEngine:
    """Main backtesting engine."""
    
    def __init__(self, data_manager: Optional[DataManager] = None):
        """
        Initialize backtesting engine.
        
        Args:
            data_manager: Data manager instance (creates new if None)
        """
        self.data_manager = data_manager or DataManager()
        self.metrics_calculator = MetricsCalculator()
    
    def run_backtest(self, 
                    trading_agent_func: Callable[[str, datetime], Tuple[str, Dict]],
                    tickers: List[str],
                    config: BacktestConfig) -> BacktestResult:
        """
        Run a complete backtest.
        
        Args:
            trading_agent_func: Function that takes (ticker, date) and returns (decision, state)
            tickers: List of tickers to trade
            config: Backtest configuration
            
        Returns:
            BacktestResult object
        """
        print(f"Starting backtest from {config.start_date} to {config.end_date}")
        print(f"Tickers: {tickers}")
        print(f"Initial cash: ${config.initial_cash:,.2f}")
        
        # Initialize portfolio
        portfolio = Portfolio(
            initial_cash=config.initial_cash,
            commission_per_trade=config.commission_per_trade
        )
        
        # Get trading days
        trading_days = self.data_manager.get_trading_days(
            config.start_date, 
            config.end_date
        )
        
        # Get benchmark data
        benchmark_data = self.data_manager.get_stock_data(
            config.benchmark_ticker,
            config.start_date,
            config.end_date
        )
        
        # Prepare data for all tickers
        ticker_data = {}
        for ticker in tickers:
            try:
                ticker_data[ticker] = self.data_manager.get_stock_data(
                    ticker, 
                    config.start_date - timedelta(days=30),  # Extra buffer for indicators
                    config.end_date + timedelta(days=config.lookback_days)
                )
            except Exception as e:
                print(f"Warning: Could not load data for {ticker}: {e}")
                continue
        
        # Track signals and decisions
        signals_log = []
        daily_decisions = {}
        
        # Main backtesting loop
        for i, current_date in enumerate(trading_days):
            print(f"Processing {current_date.strftime('%Y-%m-%d')} ({i+1}/{len(trading_days)})")
            
            # Get decisions for all tickers
            day_decisions = {}
            for ticker in tickers:
                if ticker not in ticker_data:
                    continue
                
                try:
                    # Get trading decision
                    decision, agent_state = trading_agent_func(ticker, current_date)
                    day_decisions[ticker] = decision
                    
                    # Log the signal
                    signals_log.append({
                        'date': current_date,
                        'ticker': ticker,
                        'decision': decision,
                        'agent_state_keys': list(agent_state.keys()) if agent_state else []
                    })
                    
                except Exception as e:
                    print(f"Error getting decision for {ticker} on {current_date}: {e}")
                    day_decisions[ticker] = "HOLD"
            
            daily_decisions[current_date] = day_decisions
            
            # Execute trades based on decisions
            self._execute_daily_trades(
                portfolio, 
                day_decisions, 
                current_date, 
                ticker_data, 
                config
            )
            
            # Update portfolio with current prices
            current_prices = {}
            for ticker in tickers:
                if ticker in ticker_data:
                    try:
                        price = self.data_manager.get_price_at_date(ticker, current_date)
                        current_prices[ticker] = price
                    except:
                        continue
            
            portfolio.update_prices(current_prices, current_date)
        
        # Calculate final metrics
        daily_values_df = portfolio.get_daily_values()
        trades_df = portfolio.get_trade_history()
        
        # Get benchmark returns for comparison
        benchmark_returns = benchmark_data['Close'].pct_change().dropna()
        
        # Calculate performance metrics
        if not daily_values_df.empty:
            portfolio_values = daily_values_df.set_index('Date')['Portfolio Value']
            metrics = self.metrics_calculator.calculate_metrics(
                portfolio_values, 
                trades_df, 
                benchmark_returns
            )
        else:
            # Create empty metrics if no data
            metrics = PerformanceMetrics(
                total_return=0, total_return_pct=0, annualized_return=0,
                volatility=0, sharpe_ratio=0, sortino_ratio=0,
                max_drawdown=0, max_drawdown_duration=0,
                total_trades=0, win_rate=0, avg_win=0, avg_loss=0, profit_factor=0,
                calmar_ratio=0, information_ratio=0
            )
        
        # Create result object
        result = BacktestResult(
            config=config,
            portfolio=portfolio,
            metrics=metrics,
            daily_values=daily_values_df,
            trades=trades_df,
            benchmark_data=benchmark_data,
            signals_log=signals_log
        )
        
        # Save results if requested
        if config.save_results:
            results_dir = Path(config.results_dir)
            results_dir.mkdir(parents=True, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"backtest_result_{timestamp}.json"
            result.save_to_file(results_dir / filename)
            print(f"Results saved to {results_dir / filename}")
        
        return result
    
    def _execute_daily_trades(self, 
                             portfolio: Portfolio,
                             decisions: Dict[str, str],
                             current_date: datetime,
                             ticker_data: Dict[str, pd.DataFrame],
                             config: BacktestConfig):
        """Execute trades for a single day."""
        
        for ticker, decision in decisions.items():
            if ticker not in ticker_data or decision == "HOLD":
                continue
            
            try:
                # Get current price
                current_price = self.data_manager.get_price_at_date(ticker, current_date)
                
                # Apply slippage
                if config.slippage_pct > 0:
                    slippage_multiplier = 1 + (config.slippage_pct / 100)
                    if decision == "BUY":
                        current_price *= slippage_multiplier
                    elif decision == "SELL":
                        current_price /= slippage_multiplier
                
                # Calculate position size
                shares = self._calculate_position_size(
                    portfolio, 
                    ticker, 
                    current_price, 
                    decision, 
                    config
                )
                
                if shares > 0:
                    # Execute the trade
                    action = TradeAction.BUY if decision == "BUY" else TradeAction.SELL
                    success = portfolio.execute_trade(
                        ticker=ticker,
                        action=action,
                        shares=shares,
                        price=current_price,
                        timestamp=current_date
                    )
                    
                    if success:
                        print(f"  Executed {decision}: {shares:.2f} shares of {ticker} at ${current_price:.2f}")
                    else:
                        print(f"  Failed to execute {decision} for {ticker}")
                
            except Exception as e:
                print(f"  Error executing trade for {ticker}: {e}")
    
    def _calculate_position_size(self, 
                                portfolio: Portfolio,
                                ticker: str,
                                price: float,
                                decision: str,
                                config: BacktestConfig) -> float:
        """Calculate position size based on configuration."""
        
        if decision == "SELL":
            # For sell orders, sell all shares if we have them
            if ticker in portfolio.positions:
                return portfolio.positions[ticker].shares
            else:
                return 0
        
        # For buy orders
        if config.position_size_method == "fixed_dollar":
            # Fixed dollar amount
            shares = config.position_size_value / price
            
        elif config.position_size_method == "fixed_shares":
            # Fixed number of shares
            shares = config.position_size_value
            
        elif config.position_size_method == "percent_portfolio":
            # Percentage of portfolio value
            portfolio_value = portfolio.total_value
            dollar_amount = portfolio_value * (config.position_size_value / 100)
            shares = dollar_amount / price
            
        else:
            raise ValueError(f"Unknown position sizing method: {config.position_size_method}")
        
        # Apply maximum position size constraint
        max_dollar_amount = portfolio.total_value * config.max_position_size
        max_shares = max_dollar_amount / price
        shares = min(shares, max_shares)
        
        # Ensure we don't exceed available cash
        required_cash = shares * price + config.commission_per_trade
        if required_cash > portfolio.cash:
            shares = (portfolio.cash - config.commission_per_trade) / price
        
        return max(0, shares)  # Ensure non-negative
    
    def run_walk_forward_analysis(self,
                                 trading_agent_func: Callable[[str, datetime], Tuple[str, Dict]],
                                 tickers: List[str],
                                 base_config: BacktestConfig,
                                 train_period_days: int = 252,
                                 test_period_days: int = 63) -> List[BacktestResult]:
        """
        Run walk-forward analysis.
        
        Args:
            trading_agent_func: Trading agent function
            tickers: List of tickers
            base_config: Base configuration
            train_period_days: Training period length
            test_period_days: Test period length
            
        Returns:
            List of BacktestResult objects
        """
        results = []
        current_start = base_config.start_date
        
        while current_start + timedelta(days=train_period_days + test_period_days) <= base_config.end_date:
            # Define periods
            train_start = current_start
            train_end = current_start + timedelta(days=train_period_days)
            test_start = train_end + timedelta(days=1)
            test_end = test_start + timedelta(days=test_period_days)
            
            print(f"Walk-forward period: {test_start} to {test_end}")
            
            # Create config for this period
            period_config = BacktestConfig(
                start_date=test_start,
                end_date=test_end,
                initial_cash=base_config.initial_cash,
                commission_per_trade=base_config.commission_per_trade,
                position_size_method=base_config.position_size_method,
                position_size_value=base_config.position_size_value,
                max_position_size=base_config.max_position_size,
                benchmark_ticker=base_config.benchmark_ticker,
                save_results=False  # Don't save individual results
            )
            
            # Run backtest for this period
            result = self.run_backtest(trading_agent_func, tickers, period_config)
            results.append(result)
            
            # Move to next period
            current_start = test_start
        
        return results
