"""
Data Manager for TradingAgents Evaluation Framework

Handles data fetching, caching, and management for backtesting and evaluation.
"""

import os
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import pickle


class DataManager:
    """Manages historical and real-time data for evaluation."""
    
    def __init__(self, cache_dir: str = "./evaluation/data_cache"):
        """
        Initialize DataManager.
        
        Args:
            cache_dir: Directory to store cached data
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._data_cache = {}
    
    def get_stock_data(self, 
                      ticker: str, 
                      start_date: datetime, 
                      end_date: datetime,
                      use_cache: bool = True) -> pd.DataFrame:
        """
        Get stock data for a given ticker and date range.
        
        Args:
            ticker: Stock ticker symbol
            start_date: Start date for data
            end_date: End date for data
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame with OHLCV data
        """
        cache_key = f"{ticker}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        # Try to load from cache
        if use_cache and cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Cache load failed for {ticker}: {e}")
        
        # Fetch from yfinance
        try:
            stock = yf.Ticker(ticker)
            data = stock.history(start=start_date, end=end_date + timedelta(days=1))
            
            if data.empty:
                raise ValueError(f"No data available for {ticker}")
            
            # Cache the data
            if use_cache:
                with open(cache_file, 'wb') as f:
                    pickle.dump(data, f)
            
            return data
            
        except Exception as e:
            raise ValueError(f"Failed to fetch data for {ticker}: {e}")
    
    def get_price_at_date(self, 
                         ticker: str, 
                         date: datetime, 
                         price_type: str = 'Close') -> float:
        """
        Get stock price at a specific date.
        
        Args:
            ticker: Stock ticker symbol
            date: Date to get price for
            price_type: Type of price (Open, High, Low, Close)
            
        Returns:
            Stock price
        """
        # Get data for a small window around the date
        start_date = date - timedelta(days=5)
        end_date = date + timedelta(days=5)
        
        data = self.get_stock_data(ticker, start_date, end_date)
        
        # Find the closest trading day
        target_date = date.strftime('%Y-%m-%d')
        
        if target_date in data.index.strftime('%Y-%m-%d'):
            return data.loc[data.index.strftime('%Y-%m-%d') == target_date, price_type].iloc[0]
        else:
            # Find the closest available date
            available_dates = pd.to_datetime(data.index.strftime('%Y-%m-%d'))
            closest_date_idx = (available_dates - pd.to_datetime(target_date)).abs().idxmin()
            return data.loc[closest_date_idx, price_type]
    
    def get_market_data_for_period(self, 
                                  tickers: List[str], 
                                  start_date: datetime, 
                                  end_date: datetime) -> Dict[str, pd.DataFrame]:
        """
        Get market data for multiple tickers over a period.
        
        Args:
            tickers: List of ticker symbols
            start_date: Start date
            end_date: End date
            
        Returns:
            Dictionary mapping tickers to their data
        """
        market_data = {}
        
        for ticker in tickers:
            try:
                market_data[ticker] = self.get_stock_data(ticker, start_date, end_date)
            except Exception as e:
                print(f"Failed to get data for {ticker}: {e}")
                continue
        
        return market_data
    
    def get_trading_days(self, 
                        start_date: datetime, 
                        end_date: datetime, 
                        ticker: str = "SPY") -> List[datetime]:
        """
        Get list of trading days between two dates.
        
        Args:
            start_date: Start date
            end_date: End date
            ticker: Reference ticker for trading days (default: SPY)
            
        Returns:
            List of trading days
        """
        data = self.get_stock_data(ticker, start_date, end_date)
        return [pd.to_datetime(date).to_pydatetime() for date in data.index]
    
    def validate_data_availability(self, 
                                  ticker: str, 
                                  start_date: datetime, 
                                  end_date: datetime) -> Tuple[bool, str]:
        """
        Validate if data is available for the given period.
        
        Args:
            ticker: Stock ticker
            start_date: Start date
            end_date: End date
            
        Returns:
            Tuple of (is_available, message)
        """
        try:
            data = self.get_stock_data(ticker, start_date, end_date)
            
            if data.empty:
                return False, f"No data available for {ticker}"
            
            # Check for significant gaps
            expected_days = (end_date - start_date).days
            actual_days = len(data)
            
            if actual_days < expected_days * 0.7:  # Less than 70% coverage
                return False, f"Insufficient data coverage for {ticker}: {actual_days}/{expected_days} days"
            
            return True, "Data validation successful"
            
        except Exception as e:
            return False, f"Data validation failed: {e}"
    
    def clear_cache(self, ticker: Optional[str] = None):
        """
        Clear cached data.
        
        Args:
            ticker: Specific ticker to clear, or None for all
        """
        if ticker:
            cache_files = list(self.cache_dir.glob(f"{ticker}_*.pkl"))
        else:
            cache_files = list(self.cache_dir.glob("*.pkl"))
        
        for cache_file in cache_files:
            cache_file.unlink()
        
        print(f"Cleared {len(cache_files)} cache files")
