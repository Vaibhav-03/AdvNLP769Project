"""
Visualization tools for TradingAgents Evaluation Framework

Provides comprehensive visualization capabilities for evaluation results.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Any
from pathlib import Path

from .metrics import PerformanceMetrics


class PerformanceVisualizer:
    """Creates visualizations for trading performance analysis."""
    
    def __init__(self, style: str = 'seaborn-v0_8', figsize: tuple = (12, 8)):
        """
        Initialize visualizer.
        
        Args:
            style: Matplotlib style
            figsize: Default figure size
        """
        try:
            plt.style.use(style)
        except:
            plt.style.use('default')
        
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                      '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    
    def plot_portfolio_value(self, 
                           daily_values: pd.DataFrame,
                           benchmark_data: pd.DataFrame,
                           save_path: Optional[str] = None,
                           show: bool = True) -> plt.Figure:
        """
        Plot portfolio value over time vs benchmark.
        
        Args:
            daily_values: DataFrame with Date and Portfolio Value columns
            benchmark_data: Benchmark price data
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, height_ratios=[3, 1])
        
        # Main portfolio value plot
        if not daily_values.empty:
            dates = pd.to_datetime(daily_values['Date'])
            values = daily_values['Portfolio Value']
            
            ax1.plot(dates, values, label='Portfolio', linewidth=2, color=self.colors[0])
            
            # Normalize benchmark to same starting value
            if not benchmark_data.empty:
                benchmark_dates = benchmark_data.index
                benchmark_values = benchmark_data['Close']
                
                # Align dates
                start_date = dates.iloc[0]
                end_date = dates.iloc[-1]
                
                mask = (benchmark_dates >= start_date) & (benchmark_dates <= end_date)
                aligned_benchmark = benchmark_values[mask]
                
                if len(aligned_benchmark) > 0:
                    # Normalize to same starting value
                    initial_portfolio = values.iloc[0]
                    initial_benchmark = aligned_benchmark.iloc[0]
                    normalized_benchmark = aligned_benchmark * (initial_portfolio / initial_benchmark)
                    
                    ax1.plot(aligned_benchmark.index, normalized_benchmark, 
                            label='Benchmark (SPY)', linewidth=2, color=self.colors[1], alpha=0.7)
        
        ax1.set_title('Portfolio Value Over Time', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Format x-axis
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        
        # Daily returns plot
        if not daily_values.empty and 'Daily Return' in daily_values.columns:
            returns = daily_values['Daily Return'].dropna()
            ax2.plot(dates[1:], returns, color=self.colors[2], alpha=0.7, linewidth=1)
            ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax2.set_ylabel('Daily Return', fontsize=12)
            ax2.set_xlabel('Date', fontsize=12)
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_performance_metrics(self,
                               metrics: PerformanceMetrics,
                               save_path: Optional[str] = None,
                               show: bool = True) -> plt.Figure:
        """
        Plot key performance metrics.
        
        Args:
            metrics: PerformanceMetrics object
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        
        # Return metrics
        returns_data = [
            metrics.total_return_pct,
            metrics.annualized_return
        ]
        returns_labels = ['Total Return %', 'Annualized Return %']
        
        bars1 = ax1.bar(returns_labels, returns_data, color=[self.colors[0], self.colors[1]])
        ax1.set_title('Return Metrics', fontweight='bold')
        ax1.set_ylabel('Return (%)')
        ax1.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars1, returns_data):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # Risk metrics
        risk_data = [
            metrics.volatility,
            abs(metrics.max_drawdown)
        ]
        risk_labels = ['Volatility %', 'Max Drawdown %']
        
        bars2 = ax2.bar(risk_labels, risk_data, color=[self.colors[2], self.colors[3]])
        ax2.set_title('Risk Metrics', fontweight='bold')
        ax2.set_ylabel('Percentage (%)')
        ax2.grid(True, alpha=0.3)
        
        for bar, value in zip(bars2, risk_data):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # Risk-adjusted metrics
        ratios_data = [
            metrics.sharpe_ratio,
            metrics.sortino_ratio,
            metrics.calmar_ratio
        ]
        ratios_labels = ['Sharpe', 'Sortino', 'Calmar']
        
        bars3 = ax3.bar(ratios_labels, ratios_data, color=[self.colors[4], self.colors[5], self.colors[6]])
        ax3.set_title('Risk-Adjusted Ratios', fontweight='bold')
        ax3.set_ylabel('Ratio')
        ax3.grid(True, alpha=0.3)
        
        for bar, value in zip(bars3, ratios_data):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # Trading metrics
        trading_data = [
            metrics.win_rate,
            metrics.total_trades
        ]
        trading_labels = ['Win Rate %', 'Total Trades']
        
        # Use different scales for the two metrics
        ax4_twin = ax4.twinx()
        
        bar4_1 = ax4.bar(trading_labels[0], trading_data[0], color=self.colors[7], width=0.4, alpha=0.7)
        bar4_2 = ax4_twin.bar(trading_labels[1], trading_data[1], color=self.colors[8], width=0.4, alpha=0.7)
        
        ax4.set_ylabel('Win Rate (%)', color=self.colors[7])
        ax4_twin.set_ylabel('Number of Trades', color=self.colors[8])
        ax4.set_title('Trading Activity', fontweight='bold')
        
        # Add value labels
        ax4.text(0, trading_data[0], f'{trading_data[0]:.1f}%', ha='center', va='bottom')
        ax4_twin.text(1, trading_data[1], f'{int(trading_data[1])}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_drawdown_analysis(self,
                             daily_values: pd.DataFrame,
                             save_path: Optional[str] = None,
                             show: bool = True) -> plt.Figure:
        """
        Plot drawdown analysis.
        
        Args:
            daily_values: DataFrame with portfolio values
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        if daily_values.empty:
            return plt.figure()
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize, height_ratios=[2, 1])
        
        dates = pd.to_datetime(daily_values['Date'])
        values = daily_values['Portfolio Value']
        
        # Calculate running maximum and drawdown
        running_max = values.expanding().max()
        drawdown = (values - running_max) / running_max * 100
        
        # Portfolio value with running maximum
        ax1.plot(dates, values, label='Portfolio Value', color=self.colors[0], linewidth=2)
        ax1.plot(dates, running_max, label='Running Maximum', color=self.colors[1], 
                linestyle='--', alpha=0.7)
        ax1.set_title('Portfolio Value and Running Maximum', fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Drawdown
        ax2.fill_between(dates, drawdown, 0, color=self.colors[3], alpha=0.3)
        ax2.plot(dates, drawdown, color=self.colors[3], linewidth=1)
        ax2.set_title('Drawdown Analysis', fontweight='bold')
        ax2.set_ylabel('Drawdown (%)')
        ax2.set_xlabel('Date')
        ax2.grid(True, alpha=0.3)
        
        # Highlight maximum drawdown
        max_dd_idx = drawdown.idxmin()
        max_dd_value = drawdown.min()
        ax2.scatter(dates.iloc[max_dd_idx], max_dd_value, color='red', s=100, zorder=5)
        ax2.annotate(f'Max DD: {max_dd_value:.1f}%', 
                    xy=(dates.iloc[max_dd_idx], max_dd_value),
                    xytext=(10, 10), textcoords='offset points',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_strategy_comparison(self,
                               strategies: Dict[str, pd.DataFrame],
                               save_path: Optional[str] = None,
                               show: bool = True) -> plt.Figure:
        """
        Compare multiple strategies.
        
        Args:
            strategies: Dictionary mapping strategy names to daily values DataFrames
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for i, (name, daily_values) in enumerate(strategies.items()):
            if daily_values.empty:
                continue
            
            dates = pd.to_datetime(daily_values['Date'])
            values = daily_values['Portfolio Value']
            
            # Normalize to starting value of 100
            normalized_values = (values / values.iloc[0]) * 100
            
            ax.plot(dates, normalized_values, label=name, 
                   color=self.colors[i % len(self.colors)], linewidth=2)
        
        ax.set_title('Strategy Comparison (Normalized)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Normalized Value (Starting = 100)', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_rolling_metrics(self,
                           rolling_metrics: pd.DataFrame,
                           save_path: Optional[str] = None,
                           show: bool = True) -> plt.Figure:
        """
        Plot rolling performance metrics.
        
        Args:
            rolling_metrics: DataFrame with rolling metrics
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        if rolling_metrics.empty:
            return plt.figure()
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=self.figsize)
        
        dates = rolling_metrics.index
        
        # Rolling returns
        if 'Rolling Return' in rolling_metrics.columns:
            ax1.plot(dates, rolling_metrics['Rolling Return'], color=self.colors[0])
            ax1.set_title('Rolling Returns', fontweight='bold')
            ax1.set_ylabel('Return (%)')
            ax1.grid(True, alpha=0.3)
        
        # Rolling volatility
        if 'Rolling Volatility' in rolling_metrics.columns:
            ax2.plot(dates, rolling_metrics['Rolling Volatility'], color=self.colors[1])
            ax2.set_title('Rolling Volatility', fontweight='bold')
            ax2.set_ylabel('Volatility (%)')
            ax2.grid(True, alpha=0.3)
        
        # Rolling Sharpe ratio
        if 'Rolling Sharpe' in rolling_metrics.columns:
            ax3.plot(dates, rolling_metrics['Rolling Sharpe'], color=self.colors[2])
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax3.set_title('Rolling Sharpe Ratio', fontweight='bold')
            ax3.set_ylabel('Sharpe Ratio')
            ax3.grid(True, alpha=0.3)
        
        # Rolling max drawdown
        if 'Rolling Max DD' in rolling_metrics.columns:
            ax4.plot(dates, rolling_metrics['Rolling Max DD'], color=self.colors[3])
            ax4.set_title('Rolling Max Drawdown', fontweight='bold')
            ax4.set_ylabel('Max Drawdown (%)')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def create_evaluation_dashboard(self,
                                  daily_values: pd.DataFrame,
                                  metrics: PerformanceMetrics,
                                  trades: pd.DataFrame,
                                  benchmark_data: pd.DataFrame,
                                  save_path: Optional[str] = None,
                                  show: bool = True) -> plt.Figure:
        """
        Create comprehensive evaluation dashboard.
        
        Args:
            daily_values: Portfolio daily values
            metrics: Performance metrics
            trades: Trade history
            benchmark_data: Benchmark data
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            matplotlib Figure object
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Create grid layout
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # Portfolio value (top row, spans 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        if not daily_values.empty:
            dates = pd.to_datetime(daily_values['Date'])
            values = daily_values['Portfolio Value']
            ax1.plot(dates, values, label='Portfolio', linewidth=2, color=self.colors[0])
            
            # Add benchmark if available
            if not benchmark_data.empty:
                benchmark_dates = benchmark_data.index
                benchmark_values = benchmark_data['Close']
                
                # Normalize benchmark
                if len(values) > 0 and len(benchmark_values) > 0:
                    initial_portfolio = values.iloc[0]
                    initial_benchmark = benchmark_values.iloc[0]
                    normalized_benchmark = benchmark_values * (initial_portfolio / initial_benchmark)
                    
                    ax1.plot(benchmark_dates, normalized_benchmark, 
                            label='Benchmark', linewidth=2, color=self.colors[1], alpha=0.7)
            
            ax1.set_title('Portfolio Performance', fontweight='bold')
            ax1.set_ylabel('Value ($)')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Key metrics (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        key_metrics = [
            ('Total Return', f"{metrics.total_return_pct:.1f}%"),
            ('Sharpe Ratio', f"{metrics.sharpe_ratio:.2f}"),
            ('Max Drawdown', f"{metrics.max_drawdown:.1f}%"),
            ('Win Rate', f"{metrics.win_rate:.1f}%")
        ]
        
        ax2.axis('off')
        ax2.set_title('Key Metrics', fontweight='bold', pad=20)
        
        for i, (metric, value) in enumerate(key_metrics):
            ax2.text(0.1, 0.8 - i*0.2, f"{metric}:", fontweight='bold', fontsize=12)
            ax2.text(0.6, 0.8 - i*0.2, value, fontsize=12)
        
        # Drawdown (middle left)
        ax3 = fig.add_subplot(gs[1, 0])
        if not daily_values.empty:
            values = daily_values['Portfolio Value']
            running_max = values.expanding().max()
            drawdown = (values - running_max) / running_max * 100
            
            ax3.fill_between(range(len(drawdown)), drawdown, 0, 
                           color=self.colors[3], alpha=0.3)
            ax3.plot(drawdown, color=self.colors[3])
            ax3.set_title('Drawdown', fontweight='bold')
            ax3.set_ylabel('Drawdown (%)')
            ax3.grid(True, alpha=0.3)
        
        # Monthly returns heatmap (middle center)
        ax4 = fig.add_subplot(gs[1, 1])
        if not daily_values.empty and len(daily_values) > 30:
            try:
                # Calculate monthly returns
                daily_values_copy = daily_values.copy()
                daily_values_copy['Date'] = pd.to_datetime(daily_values_copy['Date'])
                daily_values_copy.set_index('Date', inplace=True)
                
                monthly_returns = daily_values_copy['Portfolio Value'].resample('M').last().pct_change() * 100
                
                if len(monthly_returns) > 1:
                    # Create heatmap data
                    monthly_returns_df = monthly_returns.to_frame()
                    monthly_returns_df['Year'] = monthly_returns_df.index.year
                    monthly_returns_df['Month'] = monthly_returns_df.index.month
                    
                    pivot_table = monthly_returns_df.pivot(index='Year', columns='Month', 
                                                         values='Portfolio Value')
                    
                    sns.heatmap(pivot_table, annot=True, fmt='.1f', cmap='RdYlGn', 
                              center=0, ax=ax4, cbar_kws={'label': 'Return (%)'})
                    ax4.set_title('Monthly Returns Heatmap', fontweight='bold')
                else:
                    ax4.text(0.5, 0.5, 'Insufficient data\nfor monthly heatmap', 
                           ha='center', va='center', transform=ax4.transAxes)
                    ax4.set_title('Monthly Returns Heatmap', fontweight='bold')
            except Exception as e:
                ax4.text(0.5, 0.5, f'Error creating heatmap:\n{str(e)}', 
                       ha='center', va='center', transform=ax4.transAxes)
                ax4.set_title('Monthly Returns Heatmap', fontweight='bold')
        else:
            ax4.text(0.5, 0.5, 'Insufficient data\nfor monthly heatmap', 
                   ha='center', va='center', transform=ax4.transAxes)
            ax4.set_title('Monthly Returns Heatmap', fontweight='bold')
        
        # Trade distribution (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        if not trades.empty:
            trade_counts = trades['Action'].value_counts()
            ax5.pie(trade_counts.values, labels=trade_counts.index, autopct='%1.1f%%',
                   colors=[self.colors[i] for i in range(len(trade_counts))])
            ax5.set_title('Trade Distribution', fontweight='bold')
        else:
            ax5.text(0.5, 0.5, 'No trades executed', ha='center', va='center')
            ax5.set_title('Trade Distribution', fontweight='bold')
        
        # Risk metrics (bottom left)
        ax6 = fig.add_subplot(gs[2, 0])
        risk_metrics = ['Volatility', 'Sharpe Ratio', 'Sortino Ratio']
        risk_values = [metrics.volatility, metrics.sharpe_ratio, metrics.sortino_ratio]
        
        bars = ax6.bar(risk_metrics, risk_values, color=[self.colors[4], self.colors[5], self.colors[6]])
        ax6.set_title('Risk Metrics', fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        for bar, value in zip(bars, risk_values):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.2f}', ha='center', va='bottom')
        
        # Return distribution (bottom center and right)
        ax7 = fig.add_subplot(gs[2, 1:])
        if not daily_values.empty and 'Daily Return' in daily_values.columns:
            returns = daily_values['Daily Return'].dropna() * 100
            
            ax7.hist(returns, bins=30, alpha=0.7, color=self.colors[2], edgecolor='black')
            ax7.axvline(returns.mean(), color='red', linestyle='--', 
                       label=f'Mean: {returns.mean():.2f}%')
            ax7.set_title('Daily Returns Distribution', fontweight='bold')
            ax7.set_xlabel('Daily Return (%)')
            ax7.set_ylabel('Frequency')
            ax7.legend()
            ax7.grid(True, alpha=0.3)
        else:
            ax7.text(0.5, 0.5, 'No return data available', ha='center', va='center')
            ax7.set_title('Daily Returns Distribution', fontweight='bold')
        
        plt.suptitle('TradingAgents Evaluation Dashboard', fontsize=16, fontweight='bold')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
