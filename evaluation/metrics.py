"""
Performance Metrics for TradingAgents Evaluation Framework

Calculates comprehensive performance metrics for trading strategies.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from datetime import datetime
import math


@dataclass
class PerformanceMetrics:
    """Container for performance metrics."""
    
    # Return Metrics
    total_return: float
    total_return_pct: float
    annualized_return: float
    
    # Risk Metrics
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    max_drawdown: float
    max_drawdown_duration: int
    
    # Trade Metrics
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    profit_factor: float
    
    # Additional Metrics
    calmar_ratio: float
    information_ratio: float
    beta: Optional[float] = None
    alpha: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return {
            'Total Return': f"{self.total_return:.2f}",
            'Total Return %': f"{self.total_return_pct:.2f}%",
            'Annualized Return': f"{self.annualized_return:.2f}%",
            'Volatility': f"{self.volatility:.2f}%",
            'Sharpe Ratio': f"{self.sharpe_ratio:.3f}",
            'Sortino Ratio': f"{self.sortino_ratio:.3f}",
            'Max Drawdown': f"{self.max_drawdown:.2f}%",
            'Max DD Duration': f"{self.max_drawdown_duration} days",
            'Total Trades': self.total_trades,
            'Win Rate': f"{self.win_rate:.2f}%",
            'Avg Win': f"{self.avg_win:.2f}%",
            'Avg Loss': f"{self.avg_loss:.2f}%",
            'Profit Factor': f"{self.profit_factor:.3f}",
            'Calmar Ratio': f"{self.calmar_ratio:.3f}",
            'Information Ratio': f"{self.information_ratio:.3f}",
            'Beta': f"{self.beta:.3f}" if self.beta is not None else "N/A",
            'Alpha': f"{self.alpha:.3f}" if self.alpha is not None else "N/A"
        }


class MetricsCalculator:
    """Calculates performance metrics for trading strategies."""
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Initialize metrics calculator.
        
        Args:
            risk_free_rate: Annual risk-free rate (default: 2%)
        """
        self.risk_free_rate = risk_free_rate
    
    def calculate_metrics(self, 
                         portfolio_values: pd.Series,
                         trades_df: pd.DataFrame,
                         benchmark_returns: Optional[pd.Series] = None) -> PerformanceMetrics:
        """
        Calculate comprehensive performance metrics.
        
        Args:
            portfolio_values: Time series of portfolio values
            trades_df: DataFrame of trades
            benchmark_returns: Benchmark returns for alpha/beta calculation
            
        Returns:
            PerformanceMetrics object
        """
        # Calculate returns
        returns = portfolio_values.pct_change().dropna()
        
        # Return metrics
        total_return = portfolio_values.iloc[-1] - portfolio_values.iloc[0]
        total_return_pct = (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1) * 100
        
        # Annualized return
        days = len(portfolio_values)
        years = days / 252  # Assuming 252 trading days per year
        annualized_return = (((portfolio_values.iloc[-1] / portfolio_values.iloc[0]) ** (1/years)) - 1) * 100 if years > 0 else 0
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252) * 100  # Annualized volatility
        
        # Sharpe ratio
        excess_returns = returns - (self.risk_free_rate / 252)
        sharpe_ratio = (excess_returns.mean() * 252) / (returns.std() * np.sqrt(252)) if returns.std() > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0
        sortino_ratio = (excess_returns.mean() * 252) / downside_std if downside_std > 0 else 0
        
        # Drawdown metrics
        max_dd, max_dd_duration = self._calculate_drawdown_metrics(portfolio_values)
        
        # Trade metrics
        trade_metrics = self._calculate_trade_metrics(trades_df, portfolio_values)
        
        # Calmar ratio
        calmar_ratio = annualized_return / abs(max_dd) if max_dd != 0 else 0
        
        # Information ratio (vs benchmark)
        information_ratio = 0
        beta = None
        alpha = None
        
        if benchmark_returns is not None:
            # Align returns with benchmark
            aligned_returns, aligned_benchmark = returns.align(benchmark_returns, join='inner')
            
            if len(aligned_returns) > 1:
                # Information ratio
                active_returns = aligned_returns - aligned_benchmark
                tracking_error = active_returns.std() * np.sqrt(252)
                information_ratio = (active_returns.mean() * 252) / tracking_error if tracking_error > 0 else 0
                
                # Beta and Alpha
                covariance = np.cov(aligned_returns, aligned_benchmark)[0][1]
                benchmark_variance = np.var(aligned_benchmark)
                beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                
                portfolio_return = aligned_returns.mean() * 252
                benchmark_return = aligned_benchmark.mean() * 252
                alpha = portfolio_return - (self.risk_free_rate + beta * (benchmark_return - self.risk_free_rate))
        
        return PerformanceMetrics(
            total_return=total_return,
            total_return_pct=total_return_pct,
            annualized_return=annualized_return,
            volatility=volatility,
            sharpe_ratio=sharpe_ratio,
            sortino_ratio=sortino_ratio,
            max_drawdown=max_dd,
            max_drawdown_duration=max_dd_duration,
            total_trades=trade_metrics['total_trades'],
            win_rate=trade_metrics['win_rate'],
            avg_win=trade_metrics['avg_win'],
            avg_loss=trade_metrics['avg_loss'],
            profit_factor=trade_metrics['profit_factor'],
            calmar_ratio=calmar_ratio,
            information_ratio=information_ratio,
            beta=beta,
            alpha=alpha
        )
    
    def _calculate_drawdown_metrics(self, portfolio_values: pd.Series) -> Tuple[float, int]:
        """Calculate maximum drawdown and duration."""
        # Calculate running maximum
        running_max = portfolio_values.expanding().max()
        
        # Calculate drawdown
        drawdown = (portfolio_values - running_max) / running_max * 100
        
        # Maximum drawdown
        max_drawdown = drawdown.min()
        
        # Calculate drawdown duration
        max_dd_duration = 0
        current_dd_duration = 0
        
        for dd in drawdown:
            if dd < 0:
                current_dd_duration += 1
                max_dd_duration = max(max_dd_duration, current_dd_duration)
            else:
                current_dd_duration = 0
        
        return max_drawdown, max_dd_duration
    
    def _calculate_trade_metrics(self, trades_df: pd.DataFrame, portfolio_values: pd.Series) -> Dict:
        """Calculate trade-specific metrics."""
        if trades_df.empty:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0
            }
        
        # Filter out HOLD trades
        actual_trades = trades_df[trades_df['Action'] != 'HOLD'].copy()
        total_trades = len(actual_trades)
        
        if total_trades == 0:
            return {
                'total_trades': 0,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0
            }
        
        # Calculate trade returns (simplified - assumes each trade is independent)
        # For more accurate calculation, would need to track position P&L
        trade_returns = []
        
        # Group trades by ticker and calculate returns
        for ticker in actual_trades['Ticker'].unique():
            ticker_trades = actual_trades[actual_trades['Ticker'] == ticker].sort_values('Timestamp')
            
            buy_price = None
            for _, trade in ticker_trades.iterrows():
                if trade['Action'] == 'BUY':
                    buy_price = trade['Price']
                elif trade['Action'] == 'SELL' and buy_price is not None:
                    trade_return = (trade['Price'] - buy_price) / buy_price * 100
                    trade_returns.append(trade_return)
                    buy_price = None
        
        if not trade_returns:
            return {
                'total_trades': total_trades,
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'profit_factor': 0
            }
        
        # Calculate metrics
        winning_trades = [r for r in trade_returns if r > 0]
        losing_trades = [r for r in trade_returns if r < 0]
        
        win_rate = len(winning_trades) / len(trade_returns) * 100
        avg_win = np.mean(winning_trades) if winning_trades else 0
        avg_loss = np.mean(losing_trades) if losing_trades else 0
        
        total_wins = sum(winning_trades) if winning_trades else 0
        total_losses = abs(sum(losing_trades)) if losing_trades else 0
        profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0
        
        return {
            'total_trades': len(trade_returns),
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor
        }
    
    def calculate_rolling_metrics(self, 
                                 portfolio_values: pd.Series, 
                                 window: int = 252) -> pd.DataFrame:
        """
        Calculate rolling performance metrics.
        
        Args:
            portfolio_values: Time series of portfolio values
            window: Rolling window size (default: 252 trading days)
            
        Returns:
            DataFrame with rolling metrics
        """
        returns = portfolio_values.pct_change().dropna()
        
        rolling_metrics = pd.DataFrame(index=returns.index)
        
        # Rolling return
        rolling_metrics['Rolling Return'] = returns.rolling(window).apply(
            lambda x: (1 + x).prod() - 1
        ) * 100
        
        # Rolling volatility
        rolling_metrics['Rolling Volatility'] = returns.rolling(window).std() * np.sqrt(252) * 100
        
        # Rolling Sharpe ratio
        excess_returns = returns - (self.risk_free_rate / 252)
        rolling_metrics['Rolling Sharpe'] = (
            excess_returns.rolling(window).mean() * 252
        ) / (returns.rolling(window).std() * np.sqrt(252))
        
        # Rolling max drawdown
        rolling_metrics['Rolling Max DD'] = portfolio_values.rolling(window).apply(
            lambda x: self._calculate_drawdown_metrics(x)[0]
        )
        
        return rolling_metrics.dropna()
    
    def compare_strategies(self, 
                         strategies: Dict[str, pd.Series],
                         benchmark: Optional[pd.Series] = None) -> pd.DataFrame:
        """
        Compare multiple strategies.
        
        Args:
            strategies: Dictionary mapping strategy names to portfolio values
            benchmark: Benchmark portfolio values
            
        Returns:
            DataFrame comparing all strategies
        """
        comparison_data = []
        
        for name, portfolio_values in strategies.items():
            # Calculate metrics for this strategy
            trades_df = pd.DataFrame()  # Empty for now - would need actual trades
            benchmark_returns = benchmark.pct_change().dropna() if benchmark is not None else None
            
            metrics = self.calculate_metrics(portfolio_values, trades_df, benchmark_returns)
            
            comparison_data.append({
                'Strategy': name,
                **metrics.to_dict()
            })
        
        return pd.DataFrame(comparison_data)
