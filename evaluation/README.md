# TradingAgents Evaluation Framework

A comprehensive evaluation and backtesting framework for the TradingAgents multi-agent trading system.

## Overview

This framework provides complete evaluation capabilities including:
- **Historical Backtesting**: Test strategies on historical data
- **Portfolio Management**: Simulate realistic trading execution
- **Performance Metrics**: Calculate comprehensive financial metrics
- **Learning Integration**: Enable agent learning and adaptation
- **Visualization**: Generate charts and reports
- **Comparison Tools**: Compare different configurations and strategies

## Architecture

```
evaluation/
├── __init__.py              # Main exports
├── backtesting.py           # Backtesting engine
├── portfolio.py             # Portfolio management
├── metrics.py               # Performance metrics
├── evaluator.py             # Main evaluation orchestrator
├── data_manager.py          # Data fetching and caching
├── visualization.py         # Charts and plots
└── README.md               # This file
```

## Quick Start

### 1. Basic Usage

```python
from evaluation import TradingEvaluator
from datetime import datetime

# Initialize evaluator
evaluator = TradingEvaluator()

# Run evaluation
result = evaluator.evaluate_strategy(
    tickers=["AAPL", "MSFT"],
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 6, 30),
    initial_cash=100000.0
)

# View results
print(f"Total Return: {result.metrics.total_return_pct:.2f}%")
print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.3f}")
```

### 2. Configuration Comparison

```python
configs = {
    "Conservative": {"max_debate_rounds": 2},
    "Aggressive": {"max_debate_rounds": 1}
}

comparison = evaluator.compare_configurations(
    configs=configs,
    tickers=["AAPL"],
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 3, 31)
)
```

### 3. Learning Evaluation

```python
result = evaluator.evaluate_with_learning(
    tickers=["AAPL"],
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 6, 30),
    learning_frequency=15  # Update every 15 days
)
```

## Core Components

### BacktestEngine

Handles the core backtesting logic:
- Historical data processing
- Trade execution simulation
- Walk-forward analysis
- Performance tracking

```python
from evaluation import BacktestEngine, BacktestConfig

engine = BacktestEngine()
config = BacktestConfig(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 6, 30),
    initial_cash=100000.0,
    position_size_method="fixed_dollar",
    position_size_value=20000.0
)

result = engine.run_backtest(trading_function, tickers, config)
```

### Portfolio

Manages portfolio state and trade execution:
- Position tracking
- Cash management
- Trade execution
- Performance calculation

```python
from evaluation import Portfolio, TradeAction

portfolio = Portfolio(initial_cash=100000.0)

# Execute trades
portfolio.execute_trade(
    ticker="AAPL",
    action=TradeAction.BUY,
    shares=100,
    price=150.0,
    timestamp=datetime.now()
)

# Check portfolio value
print(f"Portfolio Value: ${portfolio.total_value:,.2f}")
```

### MetricsCalculator

Calculates comprehensive performance metrics:
- Return metrics (total, annualized)
- Risk metrics (volatility, drawdown, Sharpe ratio)
- Trading metrics (win rate, profit factor)
- Benchmark comparison (alpha, beta)

```python
from evaluation import MetricsCalculator

calculator = MetricsCalculator()
metrics = calculator.calculate_metrics(
    portfolio_values=daily_values,
    trades_df=trades,
    benchmark_returns=spy_returns
)
```

### DataManager

Handles data fetching and caching:
- Historical price data
- Data validation
- Caching for performance
- Trading day calculation

```python
from evaluation import DataManager

data_manager = DataManager()
data = data_manager.get_stock_data(
    ticker="AAPL",
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 6, 30)
)
```

### PerformanceVisualizer

Creates comprehensive visualizations:
- Portfolio value charts
- Performance metrics plots
- Drawdown analysis
- Strategy comparison
- Evaluation dashboards

```python
from evaluation import PerformanceVisualizer

visualizer = PerformanceVisualizer()
fig = visualizer.plot_portfolio_value(
    daily_values=result.daily_values,
    benchmark_data=result.benchmark_data
)
```

## Configuration Options

### BacktestConfig

```python
config = BacktestConfig(
    # Time period
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 6, 30),
    
    # Portfolio settings
    initial_cash=100000.0,
    commission_per_trade=0.0,
    
    # Position sizing
    position_size_method="fixed_dollar",  # or "fixed_shares", "percent_portfolio"
    position_size_value=20000.0,
    max_position_size=0.2,  # 20% max
    
    # Risk management
    stop_loss_pct=0.05,     # 5% stop loss
    take_profit_pct=0.15,   # 15% take profit
    
    # Execution
    execution_delay=0,      # Days delay
    slippage_pct=0.001,     # 0.1% slippage
    
    # Benchmark
    benchmark_ticker="SPY"
)
```

### TradingAgents Configuration

```python
trading_config = {
    "deep_think_llm": "gpt-4o-mini",
    "quick_think_llm": "gpt-4o-mini",
    "max_debate_rounds": 1,
    "max_risk_discuss_rounds": 1,
    "online_tools": True
}

evaluator = TradingEvaluator(trading_config=trading_config)
```

## Performance Metrics

### Return Metrics
- **Total Return**: Absolute dollar gain/loss
- **Total Return %**: Percentage return
- **Annualized Return**: Yearly return rate

### Risk Metrics
- **Volatility**: Annualized standard deviation
- **Sharpe Ratio**: Risk-adjusted return
- **Sortino Ratio**: Downside risk-adjusted return
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: Return/max drawdown ratio

### Trading Metrics
- **Win Rate**: Percentage of profitable trades
- **Average Win/Loss**: Average profit/loss per trade
- **Profit Factor**: Gross profit/gross loss ratio
- **Total Trades**: Number of executed trades

### Benchmark Metrics
- **Alpha**: Excess return vs benchmark
- **Beta**: Correlation with benchmark
- **Information Ratio**: Active return/tracking error

## Advanced Features

### Walk-Forward Analysis

```python
results = engine.run_walk_forward_analysis(
    trading_agent_func=trading_function,
    tickers=["AAPL"],
    base_config=config,
    train_period_days=252,  # 1 year training
    test_period_days=63     # 3 months testing
)
```

### Rolling Metrics

```python
rolling_metrics = calculator.calculate_rolling_metrics(
    portfolio_values=daily_values,
    window=252  # 1 year rolling window
)
```

### Custom Metrics

```python
class CustomMetricsCalculator(MetricsCalculator):
    def calculate_custom_metrics(self, portfolio_values, trades_df):
        # Implement custom metrics
        return custom_metrics_dict
```

## Integration with TradingAgents

The framework seamlessly integrates with TradingAgents:

```python
from tradingagents.graph.trading_graph import TradingAgentsGraph

# Create TradingAgents instance
ta = TradingAgentsGraph(
    selected_analysts=["market", "news", "social", "fundamentals"],
    config=trading_config
)

# Define trading function
def trading_function(ticker, date):
    final_state, decision = ta.propagate(ticker, date)
    processed_decision = ta.process_signal(decision)
    return processed_decision, final_state

# Run evaluation
result = evaluator.evaluate_strategy(
    tickers=["AAPL"],
    start_date=start_date,
    end_date=end_date,
    trading_function=trading_function
)
```

## Data Requirements

### Required Data
- Historical OHLCV price data
- Trading day calendar
- Benchmark data (typically SPY)

### Data Sources
- **Primary**: Yahoo Finance (via yfinance)
- **Caching**: Local file system
- **Validation**: Automatic data quality checks

### Data Management
```python
# Validate data availability
is_valid, message = data_manager.validate_data_availability(
    ticker="AAPL",
    start_date=start_date,
    end_date=end_date
)

# Clear cache if needed
data_manager.clear_cache(ticker="AAPL")
```

## Error Handling

The framework includes comprehensive error handling:
- Data availability validation
- Trade execution validation
- Memory management for large datasets
- Graceful degradation for missing data

## Performance Optimization

### Caching
- Automatic data caching
- Configurable cache directory
- Cache validation and cleanup

### Memory Management
- Efficient data structures
- Streaming for large datasets
- Garbage collection optimization

### Parallel Processing
- Multi-ticker evaluation support
- Concurrent data fetching
- Parallel metric calculation

## Best Practices

1. **Data Validation**: Always validate data before evaluation
2. **Realistic Parameters**: Use appropriate commissions and slippage
3. **Sufficient History**: Ensure adequate data for meaningful results
4. **Benchmark Comparison**: Always compare against relevant benchmarks
5. **Risk Management**: Implement appropriate position sizing and risk controls
6. **Documentation**: Save configurations and results for reproducibility

## Troubleshooting

### Common Issues

1. **Missing Data**: Check data availability and date ranges
2. **API Limits**: Use caching and reduce evaluation frequency
3. **Memory Issues**: Use smaller datasets or streaming
4. **Performance**: Enable caching and use efficient configurations

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Use debug mode in TradingAgents
ta = TradingAgentsGraph(debug=True, config=config)
```

## Examples

See the `examples/` directory for comprehensive usage examples:
- `basic_evaluation.py`: Simple evaluation workflow
- `advanced_evaluation.py`: Advanced features and comparisons
- `custom_evaluation.py`: Custom extensions and metrics

## API Reference

### TradingEvaluator
Main evaluation orchestrator class.

**Methods**:
- `evaluate_strategy()`: Run single evaluation
- `evaluate_with_learning()`: Evaluation with agent learning
- `compare_configurations()`: Compare multiple configs
- `generate_report()`: Create comprehensive report

### BacktestEngine
Core backtesting functionality.

**Methods**:
- `run_backtest()`: Execute backtest
- `run_walk_forward_analysis()`: Time-series cross-validation

### Portfolio
Portfolio management and tracking.

**Methods**:
- `execute_trade()`: Execute a trade
- `update_prices()`: Update position values
- `get_position_summary()`: Position overview
- `get_trade_history()`: Trade history

### MetricsCalculator
Performance metrics calculation.

**Methods**:
- `calculate_metrics()`: Comprehensive metrics
- `calculate_rolling_metrics()`: Rolling window metrics
- `compare_strategies()`: Multi-strategy comparison

## Contributing

To contribute to the evaluation framework:
1. Follow existing code patterns
2. Add comprehensive tests
3. Update documentation
4. Include usage examples
5. Ensure backward compatibility

## License

This evaluation framework is part of the TradingAgents project and follows the same license terms.
