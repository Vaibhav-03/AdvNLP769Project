# TradingAgents Evaluation Examples

This directory contains comprehensive examples demonstrating how to use the TradingAgents evaluation framework.

## Overview

The evaluation framework provides tools for:
- **Backtesting**: Historical performance evaluation
- **Portfolio Management**: Position tracking and trade execution simulation
- **Performance Metrics**: Comprehensive financial metrics calculation
- **Visualization**: Charts and reports for analysis
- **Learning Integration**: Adaptive agent improvement

## Examples

### 1. Basic Evaluation (`basic_evaluation.py`)

**Purpose**: Simple introduction to the evaluation framework

**Features**:
- Single-period backtesting
- Basic performance metrics
- Report generation
- Portfolio analysis

**Usage**:
```bash
python examples/basic_evaluation.py
```

**What it demonstrates**:
- How to set up a basic evaluation
- Running TradingAgents on historical data
- Interpreting performance results
- Generating evaluation reports

### 2. Advanced Evaluation (`advanced_evaluation.py`)

**Purpose**: Advanced evaluation techniques and comparisons

**Features**:
- Configuration comparison
- Walk-forward analysis
- Learning-enabled evaluation
- Agent performance analysis

**Usage**:
```bash
python examples/advanced_evaluation.py
```

**What it demonstrates**:
- Comparing different TradingAgents configurations
- Time-series cross-validation with walk-forward analysis
- Incorporating agent learning and memory updates
- Analyzing individual agent contributions

### 3. Custom Evaluation (`custom_evaluation.py`)

**Purpose**: Extending the framework with custom functionality

**Features**:
- Custom position sizing strategies
- Advanced risk management rules
- Performance attribution analysis
- Custom metrics calculation

**Usage**:
```bash
python examples/custom_evaluation.py
```

**What it demonstrates**:
- Creating custom position sizing algorithms
- Implementing sophisticated risk management
- Building custom performance metrics
- Extending the evaluation framework

## Quick Start

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install matplotlib seaborn  # For visualization
   ```

2. **Set up API Keys**:
   ```bash
   export OPENAI_API_KEY="your_openai_key"
   export FINNHUB_API_KEY="your_finnhub_key"
   ```

3. **Run Basic Example**:
   ```bash
   python examples/basic_evaluation.py
   ```

## Configuration Options

### TradingAgents Configuration
```python
trading_config = {
    "deep_think_llm": "gpt-4o-mini",      # Model for complex analysis
    "quick_think_llm": "gpt-4o-mini",     # Model for quick decisions
    "max_debate_rounds": 1,                # Researcher debate rounds
    "max_risk_discuss_rounds": 1,          # Risk analysis rounds
    "online_tools": True                   # Use real-time data
}
```

### Backtest Configuration
```python
backtest_config = BacktestConfig(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 6, 30),
    initial_cash=100000.0,
    commission_per_trade=0.0,
    position_size_method="fixed_dollar",   # "fixed_dollar", "fixed_shares", "percent_portfolio"
    position_size_value=20000.0,
    max_position_size=0.2,                 # 20% max position size
    benchmark_ticker="SPY"
)
```

## Key Metrics Explained

### Return Metrics
- **Total Return**: Absolute dollar gain/loss
- **Total Return %**: Percentage gain/loss
- **Annualized Return**: Yearly return rate

### Risk Metrics
- **Volatility**: Standard deviation of returns
- **Maximum Drawdown**: Largest peak-to-trough decline
- **Sharpe Ratio**: Risk-adjusted return measure
- **Sortino Ratio**: Downside risk-adjusted return

### Trading Metrics
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profits to gross losses
- **Total Trades**: Number of executed trades

## Evaluation Workflow

1. **Data Preparation**:
   - Historical price data fetching
   - Data validation and caching
   - Trading day identification

2. **Strategy Execution**:
   - TradingAgents decision generation
   - Trade execution simulation
   - Portfolio tracking

3. **Performance Analysis**:
   - Metrics calculation
   - Benchmark comparison
   - Risk assessment

4. **Learning Integration**:
   - Performance feedback to agents
   - Memory updates
   - Strategy adaptation

5. **Reporting**:
   - Comprehensive reports
   - Visualization generation
   - Results storage

## Extending the Framework

### Custom Metrics
```python
class CustomMetricsCalculator(MetricsCalculator):
    def calculate_custom_metrics(self, portfolio_values, trades_df):
        # Implement your custom metrics
        return custom_metrics_dict
```

### Custom Risk Management
```python
class RiskManagedPortfolio(Portfolio):
    def update_prices(self, prices, timestamp):
        super().update_prices(prices, timestamp)
        # Implement custom risk rules
        self._apply_risk_management(prices, timestamp)
```

### Custom Position Sizing
```python
def custom_position_size(portfolio, ticker, price, decision, config):
    # Implement your position sizing logic
    return calculated_shares
```

## Best Practices

1. **Start Simple**: Begin with basic evaluation before adding complexity
2. **Use Realistic Parameters**: Set appropriate commission and slippage
3. **Validate Data**: Ensure data quality and completeness
4. **Monitor Performance**: Track key metrics throughout evaluation
5. **Document Results**: Save configurations and results for comparison
6. **Test Robustness**: Use walk-forward analysis and different time periods

## Troubleshooting

### Common Issues

1. **API Rate Limits**: Use caching and reduce evaluation frequency
2. **Data Gaps**: Validate data availability before evaluation
3. **Memory Issues**: Use smaller time periods or fewer tickers
4. **Model Costs**: Use cheaper models for testing (gpt-4o-mini)

### Performance Optimization

1. **Caching**: Enable data caching for repeated evaluations
2. **Parallel Processing**: Evaluate multiple tickers in parallel
3. **Reduced Complexity**: Lower debate rounds for faster execution
4. **Batch Processing**: Group evaluations by time period

## Support

For questions or issues:
1. Check the main README.md for setup instructions
2. Review the evaluation framework documentation
3. Examine the example code for usage patterns
4. Test with smaller datasets first

## Contributing

To contribute new examples or improvements:
1. Follow the existing code structure
2. Add comprehensive documentation
3. Include error handling
4. Test with different configurations
5. Update this README with new examples
