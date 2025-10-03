"""
Custom Evaluation Example for TradingAgents

This example shows how to create custom evaluation workflows and metrics.
"""

import sys
import os
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add the parent directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from evaluation import (
    TradingEvaluator, BacktestEngine, BacktestConfig, 
    Portfolio, DataManager, MetricsCalculator
)
from tradingagents.graph.trading_graph import TradingAgentsGraph
from tradingagents.default_config import DEFAULT_CONFIG


class CustomMetricsCalculator(MetricsCalculator):
    """Extended metrics calculator with custom metrics."""
    
    def calculate_custom_metrics(self, portfolio_values, trades_df):
        """Calculate custom performance metrics."""
        
        custom_metrics = {}
        
        if not portfolio_values.empty:
            returns = portfolio_values.pct_change().dropna()
            
            # Custom Metric 1: Ulcer Index (alternative to max drawdown)
            running_max = portfolio_values.expanding().max()
            drawdown = (portfolio_values - running_max) / running_max
            squared_drawdown = drawdown ** 2
            ulcer_index = np.sqrt(squared_drawdown.mean()) * 100
            custom_metrics['ulcer_index'] = ulcer_index
            
            # Custom Metric 2: Tail Ratio (95th percentile / 5th percentile)
            if len(returns) > 20:
                tail_ratio = returns.quantile(0.95) / abs(returns.quantile(0.05))
                custom_metrics['tail_ratio'] = tail_ratio
            else:
                custom_metrics['tail_ratio'] = 0
            
            # Custom Metric 3: Consistency Score (percentage of positive months)
            if len(portfolio_values) > 30:
                monthly_returns = portfolio_values.resample('M').last().pct_change().dropna()
                consistency_score = (monthly_returns > 0).mean() * 100
                custom_metrics['consistency_score'] = consistency_score
            else:
                custom_metrics['consistency_score'] = 0
        
        # Custom Metric 4: Trade Efficiency (if trades available)
        if not trades_df.empty:
            actual_trades = trades_df[trades_df['Action'] != 'HOLD']
            if len(actual_trades) > 0:
                # Calculate average time between trades
                actual_trades['Timestamp'] = pd.to_datetime(actual_trades['Timestamp'])
                time_diffs = actual_trades['Timestamp'].diff().dt.days.dropna()
                avg_time_between_trades = time_diffs.mean() if len(time_diffs) > 0 else 0
                custom_metrics['avg_days_between_trades'] = avg_time_between_trades
                
                # Calculate trade size consistency
                trade_values = actual_trades['Shares'] * actual_trades['Price']
                trade_size_cv = trade_values.std() / trade_values.mean() if trade_values.mean() > 0 else 0
                custom_metrics['trade_size_consistency'] = 1 / (1 + trade_size_cv)  # Higher is more consistent
            else:
                custom_metrics['avg_days_between_trades'] = 0
                custom_metrics['trade_size_consistency'] = 0
        
        return custom_metrics


def custom_position_sizing_strategy():
    """Example of custom position sizing based on volatility."""
    
    print("Custom Position Sizing Strategy Example")
    print("=" * 50)
    
    # Initialize components
    data_manager = DataManager()
    
    def volatility_based_position_size(ticker, current_date, portfolio_value, base_position_size=20000):
        """Calculate position size based on recent volatility."""
        
        try:
            # Get recent price data for volatility calculation
            end_date = current_date
            start_date = current_date - timedelta(days=30)
            
            price_data = data_manager.get_stock_data(ticker, start_date, end_date)
            
            if len(price_data) < 10:
                return base_position_size
            
            # Calculate 20-day volatility
            returns = price_data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Annualized volatility
            
            # Adjust position size inversely to volatility
            # Higher volatility = smaller position size
            volatility_adjustment = 1 / (1 + volatility * 2)  # Scale factor
            adjusted_size = base_position_size * volatility_adjustment
            
            # Cap at 10% of portfolio
            max_size = portfolio_value * 0.1
            final_size = min(adjusted_size, max_size)
            
            return final_size
            
        except Exception as e:
            print(f"Error calculating volatility-based position size for {ticker}: {e}")
            return base_position_size
    
    # Create custom backtest with volatility-based sizing
    class CustomBacktestEngine(BacktestEngine):
        """Custom backtest engine with volatility-based position sizing."""
        
        def _calculate_position_size(self, portfolio, ticker, price, decision, config):
            """Override position sizing with volatility-based approach."""
            
            if decision == "SELL":
                if ticker in portfolio.positions:
                    return portfolio.positions[ticker].shares
                else:
                    return 0
            
            # Get current date from config (would need to be passed)
            # For this example, we'll use a simplified approach
            base_size = config.position_size_value
            
            # Simple volatility adjustment (in real implementation, would use actual volatility)
            # This is a placeholder - you'd implement the actual volatility calculation
            volatility_factor = 0.8  # Assume moderate volatility
            adjusted_size = base_size * volatility_factor
            
            shares = adjusted_size / price
            
            # Apply constraints
            max_dollar_amount = portfolio.total_value * config.max_position_size
            max_shares = max_dollar_amount / price
            shares = min(shares, max_shares)
            
            required_cash = shares * price + config.commission_per_trade
            if required_cash > portfolio.cash:
                shares = (portfolio.cash - config.commission_per_trade) / price
            
            return max(0, shares)
    
    # Test the custom strategy
    tickers = ["AAPL"]
    start_date = datetime(2024, 2, 1)
    end_date = datetime(2024, 4, 30)
    
    print(f"Testing volatility-based position sizing...")
    print(f"Ticker: {tickers[0]}")
    print(f"Period: {start_date} to {end_date}")
    print()
    
    try:
        # Create TradingAgents instance
        trading_config = DEFAULT_CONFIG.copy()
        trading_config.update({
            "deep_think_llm": "gpt-4o-mini",
            "quick_think_llm": "gpt-4o-mini",
            "max_debate_rounds": 1
        })
        
        trading_graph = TradingAgentsGraph(
            selected_analysts=["market"],
            debug=False,
            config=trading_config
        )
        
        def trading_agent_func(ticker, trade_date):
            try:
                final_state, decision = trading_graph.propagate(ticker, trade_date.strftime('%Y-%m-%d'))
                processed_decision = trading_graph.process_signal(decision)
                return processed_decision, final_state
            except:
                return "HOLD", {}
        
        # Run with custom engine
        custom_engine = CustomBacktestEngine(data_manager)
        
        config = BacktestConfig(
            start_date=start_date,
            end_date=end_date,
            initial_cash=100000.0,
            position_size_method="fixed_dollar",
            position_size_value=20000.0,
            max_position_size=0.15  # 15% max position size
        )
        
        result = custom_engine.run_backtest(trading_agent_func, tickers, config)
        
        print("CUSTOM POSITION SIZING RESULTS:")
        print("-" * 40)
        print(f"Final Value: ${result.portfolio.total_value:,.2f}")
        print(f"Total Return: {result.metrics.total_return_pct:.2f}%")
        print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.3f}")
        print(f"Max Drawdown: {result.metrics.max_drawdown:.2f}%")
        
    except Exception as e:
        print(f"Error in custom position sizing test: {e}")
        import traceback
        traceback.print_exc()


def custom_risk_management():
    """Example of custom risk management rules."""
    
    print("\n" + "=" * 50)
    print("Custom Risk Management Example")
    print("=" * 50)
    
    class RiskManagedPortfolio(Portfolio):
        """Portfolio with custom risk management rules."""
        
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.stop_loss_pct = 0.05  # 5% stop loss
            self.take_profit_pct = 0.15  # 15% take profit
            self.max_portfolio_drawdown = 0.10  # 10% max portfolio drawdown
            self.peak_value = self.initial_cash
        
        def update_prices(self, prices, timestamp):
            """Override to include risk management checks."""
            super().update_prices(prices, timestamp)
            
            # Update peak value
            if self.total_value > self.peak_value:
                self.peak_value = self.total_value
            
            # Check portfolio-level stop loss
            portfolio_drawdown = (self.peak_value - self.total_value) / self.peak_value
            if portfolio_drawdown > self.max_portfolio_drawdown:
                print(f"Portfolio drawdown limit hit: {portfolio_drawdown:.2%}")
                self._liquidate_all_positions(prices, timestamp)
            
            # Check individual position stop losses and take profits
            positions_to_close = []
            
            for ticker, position in self.positions.items():
                if ticker in prices:
                    current_price = prices[ticker]
                    
                    # Calculate position P&L
                    position_pnl_pct = (current_price - position.avg_cost) / position.avg_cost
                    
                    # Check stop loss
                    if position_pnl_pct <= -self.stop_loss_pct:
                        print(f"Stop loss triggered for {ticker}: {position_pnl_pct:.2%}")
                        positions_to_close.append((ticker, current_price, "STOP_LOSS"))
                    
                    # Check take profit
                    elif position_pnl_pct >= self.take_profit_pct:
                        print(f"Take profit triggered for {ticker}: {position_pnl_pct:.2%}")
                        positions_to_close.append((ticker, current_price, "TAKE_PROFIT"))
            
            # Close positions that hit risk limits
            for ticker, price, reason in positions_to_close:
                shares = self.positions[ticker].shares
                self.execute_trade(ticker, "SELL", shares, price, timestamp)
                print(f"Closed {ticker} position: {reason}")
        
        def _liquidate_all_positions(self, prices, timestamp):
            """Liquidate all positions due to portfolio risk limit."""
            for ticker, position in list(self.positions.items()):
                if ticker in prices:
                    shares = position.shares
                    price = prices[ticker]
                    self.execute_trade(ticker, "SELL", shares, price, timestamp)
                    print(f"Liquidated {ticker} due to portfolio risk limit")
    
    print("This example shows how to implement custom risk management:")
    print("- Individual position stop losses (5%)")
    print("- Individual position take profits (15%)")
    print("- Portfolio-level drawdown limit (10%)")
    print()
    print("To use this in a backtest, you would:")
    print("1. Replace the standard Portfolio with RiskManagedPortfolio")
    print("2. The risk rules would automatically trigger during price updates")
    print("3. Monitor the additional risk management logs")


def custom_performance_attribution():
    """Example of performance attribution analysis."""
    
    print("\n" + "=" * 50)
    print("Custom Performance Attribution Example")
    print("=" * 50)
    
    def analyze_performance_attribution(trades_df, price_data_dict):
        """Analyze which trades/decisions contributed most to performance."""
        
        if trades_df.empty:
            print("No trades to analyze")
            return
        
        attribution_results = []
        
        # Group trades by ticker
        for ticker in trades_df['Ticker'].unique():
            ticker_trades = trades_df[trades_df['Ticker'] == ticker].sort_values('Timestamp')
            
            # Calculate trade-by-trade P&L
            buy_price = None
            for _, trade in ticker_trades.iterrows():
                if trade['Action'] == 'BUY':
                    buy_price = trade['Price']
                elif trade['Action'] == 'SELL' and buy_price is not None:
                    trade_pnl = (trade['Price'] - buy_price) / buy_price * 100
                    trade_value = trade['Shares'] * buy_price
                    
                    attribution_results.append({
                        'Ticker': ticker,
                        'Buy Date': ticker_trades[ticker_trades['Action'] == 'BUY']['Timestamp'].iloc[-1],
                        'Sell Date': trade['Timestamp'],
                        'Buy Price': buy_price,
                        'Sell Price': trade['Price'],
                        'Shares': trade['Shares'],
                        'Trade Value': trade_value,
                        'P&L %': trade_pnl,
                        'P&L $': trade_pnl / 100 * trade_value
                    })
                    buy_price = None
        
        if attribution_results:
            attribution_df = pd.DataFrame(attribution_results)
            
            print("PERFORMANCE ATTRIBUTION ANALYSIS:")
            print("-" * 40)
            
            # Top contributing trades
            top_trades = attribution_df.nlargest(5, 'P&L $')
            print("\nTop 5 Contributing Trades:")
            for _, trade in top_trades.iterrows():
                print(f"  {trade['Ticker']}: ${trade['P&L $']:,.0f} "
                      f"({trade['P&L %']:.1f}%) from {trade['Buy Date'].strftime('%Y-%m-%d')} "
                      f"to {trade['Sell Date'].strftime('%Y-%m-%d')}")
            
            # Worst trades
            worst_trades = attribution_df.nsmallest(5, 'P&L $')
            print("\nWorst 5 Trades:")
            for _, trade in worst_trades.iterrows():
                print(f"  {trade['Ticker']}: ${trade['P&L $']:,.0f} "
                      f"({trade['P&L %']:.1f}%) from {trade['Buy Date'].strftime('%Y-%m-%d')} "
                      f"to {trade['Sell Date'].strftime('%Y-%m-%d')}")
            
            # Summary by ticker
            ticker_summary = attribution_df.groupby('Ticker').agg({
                'P&L $': ['sum', 'count', 'mean'],
                'P&L %': 'mean'
            }).round(2)
            
            print("\nSummary by Ticker:")
            print(ticker_summary)
        
        return attribution_results
    
    # Example usage
    print("This function analyzes trade-by-trade performance attribution:")
    print("- Identifies best and worst performing trades")
    print("- Shows contribution by ticker")
    print("- Calculates average trade performance")
    print()
    print("To use this analysis:")
    print("1. Run a backtest to get trades_df")
    print("2. Call analyze_performance_attribution(trades_df, price_data)")
    print("3. Review the attribution results")


def main():
    """Run custom evaluation examples."""
    
    print("TradingAgents Custom Evaluation Examples")
    print("=" * 60)
    
    try:
        custom_position_sizing_strategy()
        custom_risk_management()
        custom_performance_attribution()
        
        print("\n" + "=" * 60)
        print("Custom evaluation examples completed!")
        print("\nThese examples show how to extend the evaluation framework:")
        print("- Custom position sizing strategies")
        print("- Advanced risk management rules")
        print("- Detailed performance attribution")
        print("- Custom metrics and analysis")
        
    except Exception as e:
        print(f"Error in custom evaluation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
