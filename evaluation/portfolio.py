"""
Portfolio Management for TradingAgents Evaluation Framework

Handles portfolio tracking, position management, and trade execution simulation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from enum import Enum
import pandas as pd


class TradeAction(Enum):
    """Trade action types."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"


@dataclass
class Trade:
    """Represents a single trade."""
    timestamp: datetime
    ticker: str
    action: TradeAction
    shares: float
    price: float
    commission: float = 0.0
    
    @property
    def total_value(self) -> float:
        """Total value of the trade including commission."""
        return (self.shares * self.price) + self.commission
    
    @property
    def net_value(self) -> float:
        """Net value considering buy/sell direction."""
        multiplier = 1 if self.action == TradeAction.BUY else -1
        return multiplier * self.total_value


@dataclass
class Position:
    """Represents a position in a security."""
    ticker: str
    shares: float
    avg_cost: float
    last_price: float = 0.0
    
    @property
    def market_value(self) -> float:
        """Current market value of position."""
        return self.shares * self.last_price
    
    @property
    def cost_basis(self) -> float:
        """Total cost basis of position."""
        return self.shares * self.avg_cost
    
    @property
    def unrealized_pnl(self) -> float:
        """Unrealized profit/loss."""
        return self.market_value - self.cost_basis
    
    @property
    def unrealized_pnl_pct(self) -> float:
        """Unrealized profit/loss percentage."""
        if self.cost_basis == 0:
            return 0.0
        return (self.unrealized_pnl / self.cost_basis) * 100


class Portfolio:
    """Manages a trading portfolio with positions and cash."""
    
    def __init__(self, 
                 initial_cash: float = 100000.0,
                 commission_per_trade: float = 0.0):
        """
        Initialize portfolio.
        
        Args:
            initial_cash: Starting cash amount
            commission_per_trade: Commission charged per trade
        """
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.commission_per_trade = commission_per_trade
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.daily_values: List[Tuple[datetime, float]] = []
    
    def execute_trade(self, 
                     ticker: str, 
                     action: TradeAction, 
                     shares: float, 
                     price: float,
                     timestamp: datetime) -> bool:
        """
        Execute a trade.
        
        Args:
            ticker: Stock ticker
            action: Trade action (BUY/SELL/HOLD)
            shares: Number of shares
            price: Price per share
            timestamp: Trade timestamp
            
        Returns:
            True if trade was executed successfully
        """
        if action == TradeAction.HOLD:
            return True
        
        trade_value = shares * price
        commission = self.commission_per_trade
        total_cost = trade_value + commission
        
        if action == TradeAction.BUY:
            # Check if we have enough cash
            if self.cash < total_cost:
                print(f"Insufficient cash for {ticker} BUY: need ${total_cost:.2f}, have ${self.cash:.2f}")
                return False
            
            # Execute buy
            self.cash -= total_cost
            
            if ticker in self.positions:
                # Update existing position
                pos = self.positions[ticker]
                total_shares = pos.shares + shares
                total_cost_basis = pos.cost_basis + trade_value
                pos.shares = total_shares
                pos.avg_cost = total_cost_basis / total_shares if total_shares > 0 else 0
            else:
                # Create new position
                self.positions[ticker] = Position(
                    ticker=ticker,
                    shares=shares,
                    avg_cost=price,
                    last_price=price
                )
        
        elif action == TradeAction.SELL:
            # Check if we have enough shares
            if ticker not in self.positions or self.positions[ticker].shares < shares:
                available_shares = self.positions.get(ticker, Position(ticker, 0, 0)).shares
                print(f"Insufficient shares for {ticker} SELL: need {shares}, have {available_shares}")
                return False
            
            # Execute sell
            self.cash += trade_value - commission
            
            pos = self.positions[ticker]
            pos.shares -= shares
            
            # Remove position if no shares left
            if pos.shares <= 0:
                del self.positions[ticker]
        
        # Record the trade
        trade = Trade(
            timestamp=timestamp,
            ticker=ticker,
            action=action,
            shares=shares,
            price=price,
            commission=commission
        )
        self.trades.append(trade)
        
        return True
    
    def update_prices(self, prices: Dict[str, float], timestamp: datetime):
        """
        Update current prices for all positions.
        
        Args:
            prices: Dictionary mapping tickers to current prices
            timestamp: Current timestamp
        """
        for ticker, position in self.positions.items():
            if ticker in prices:
                position.last_price = prices[ticker]
        
        # Record daily portfolio value
        self.daily_values.append((timestamp, self.total_value))
    
    @property
    def total_value(self) -> float:
        """Total portfolio value (cash + positions)."""
        positions_value = sum(pos.market_value for pos in self.positions.values())
        return self.cash + positions_value
    
    @property
    def total_return(self) -> float:
        """Total return since inception."""
        return self.total_value - self.initial_cash
    
    @property
    def total_return_pct(self) -> float:
        """Total return percentage."""
        return (self.total_return / self.initial_cash) * 100
    
    def get_position_summary(self) -> pd.DataFrame:
        """Get summary of current positions."""
        if not self.positions:
            return pd.DataFrame()
        
        data = []
        for ticker, pos in self.positions.items():
            data.append({
                'Ticker': ticker,
                'Shares': pos.shares,
                'Avg Cost': pos.avg_cost,
                'Last Price': pos.last_price,
                'Market Value': pos.market_value,
                'Unrealized P&L': pos.unrealized_pnl,
                'Unrealized P&L %': pos.unrealized_pnl_pct
            })
        
        return pd.DataFrame(data)
    
    def get_trade_history(self) -> pd.DataFrame:
        """Get trade history as DataFrame."""
        if not self.trades:
            return pd.DataFrame()
        
        data = []
        for trade in self.trades:
            data.append({
                'Timestamp': trade.timestamp,
                'Ticker': trade.ticker,
                'Action': trade.action.value,
                'Shares': trade.shares,
                'Price': trade.price,
                'Total Value': trade.total_value,
                'Commission': trade.commission
            })
        
        return pd.DataFrame(data)
    
    def get_daily_values(self) -> pd.DataFrame:
        """Get daily portfolio values as DataFrame."""
        if not self.daily_values:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.daily_values, columns=['Date', 'Portfolio Value'])
        df['Daily Return'] = df['Portfolio Value'].pct_change()
        df['Cumulative Return'] = (df['Portfolio Value'] / self.initial_cash - 1) * 100
        
        return df


class PortfolioManager:
    """Manages multiple portfolios and strategies."""
    
    def __init__(self):
        """Initialize portfolio manager."""
        self.portfolios: Dict[str, Portfolio] = {}
    
    def create_portfolio(self, 
                        name: str, 
                        initial_cash: float = 100000.0,
                        commission_per_trade: float = 0.0) -> Portfolio:
        """
        Create a new portfolio.
        
        Args:
            name: Portfolio name
            initial_cash: Initial cash amount
            commission_per_trade: Commission per trade
            
        Returns:
            Created portfolio
        """
        portfolio = Portfolio(initial_cash, commission_per_trade)
        self.portfolios[name] = portfolio
        return portfolio
    
    def get_portfolio(self, name: str) -> Optional[Portfolio]:
        """Get portfolio by name."""
        return self.portfolios.get(name)
    
    def compare_portfolios(self) -> pd.DataFrame:
        """Compare performance across portfolios."""
        if not self.portfolios:
            return pd.DataFrame()
        
        data = []
        for name, portfolio in self.portfolios.items():
            data.append({
                'Portfolio': name,
                'Initial Cash': portfolio.initial_cash,
                'Current Value': portfolio.total_value,
                'Total Return': portfolio.total_return,
                'Total Return %': portfolio.total_return_pct,
                'Cash': portfolio.cash,
                'Positions': len(portfolio.positions),
                'Trades': len(portfolio.trades)
            })
        
        return pd.DataFrame(data)
    
    def get_all_daily_values(self) -> pd.DataFrame:
        """Get daily values for all portfolios."""
        all_data = {}
        
        for name, portfolio in self.portfolios.items():
            daily_df = portfolio.get_daily_values()
            if not daily_df.empty:
                all_data[name] = daily_df.set_index('Date')['Portfolio Value']
        
        if not all_data:
            return pd.DataFrame()
        
        combined_df = pd.DataFrame(all_data)
        return combined_df
