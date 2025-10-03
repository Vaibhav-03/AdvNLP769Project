"""
Financial Knowledge Base for RAG

Manages financial documents, market data, and trading knowledge for RAG retrieval.
Integrates with existing data sources and provides document ingestion capabilities.
"""

import os
import json
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import yfinance as yf

from .document_store import DocumentStore
from .retriever import RAGRetriever
from .generator import RAGGenerator
from .rag_pipeline import RAGPipeline


class FinancialKnowledgeBase:
    """Financial knowledge base for RAG integration."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize financial knowledge base.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize components
        self.document_store = DocumentStore(config, "financial_knowledge_base")
        self.retriever = RAGRetriever(self.document_store, "hybrid")
        self.generator = RAGGenerator(config.get("llm"), config)
        self.pipeline = RAGPipeline(
            self.document_store, 
            self.retriever, 
            self.generator, 
            config
        )
        
        # Data sources
        self.data_cache_dir = Path(config.get("data_cache_dir", "./data_cache"))
        self.data_cache_dir.mkdir(parents=True, exist_ok=True)
    
    def ingest_financial_documents(self, 
                                  documents: List[Dict[str, Any]]) -> List[str]:
        """
        Ingest financial documents into the knowledge base.
        
        Args:
            documents: List of documents with 'content', 'metadata', and optional 'id'
            
        Returns:
            List of document IDs
        """
        return self.document_store.add_documents_batch(documents)
    
    def ingest_earnings_calls(self, 
                             ticker: str, 
                             earnings_data: List[Dict[str, Any]]) -> List[str]:
        """
        Ingest earnings call transcripts.
        
        Args:
            ticker: Stock ticker symbol
            earnings_data: List of earnings call data
            
        Returns:
            List of document IDs
        """
        documents = []
        for call in earnings_data:
            content = f"Earnings Call Transcript for {ticker}\n"
            content += f"Date: {call.get('date', 'Unknown')}\n"
            content += f"Quarter: {call.get('quarter', 'Unknown')}\n\n"
            content += call.get('transcript', '')
            
            metadata = {
                'ticker': ticker,
                'document_type': 'earnings_call',
                'date': call.get('date', ''),
                'quarter': call.get('quarter', ''),
                'source': 'earnings_calls'
            }
            
            documents.append({
                'content': content,
                'metadata': metadata
            })
        
        return self.ingest_financial_documents(documents)
    
    def ingest_sec_filings(self, 
                          ticker: str, 
                          filings_data: List[Dict[str, Any]]) -> List[str]:
        """
        Ingest SEC filings (10-K, 10-Q, etc.).
        
        Args:
            ticker: Stock ticker symbol
            filings_data: List of SEC filing data
            
        Returns:
            List of document IDs
        """
        documents = []
        for filing in filings_data:
            content = f"SEC Filing for {ticker}\n"
            content += f"Type: {filing.get('type', 'Unknown')}\n"
            content += f"Date: {filing.get('date', 'Unknown')}\n"
            content += f"Period: {filing.get('period', 'Unknown')}\n\n"
            content += filing.get('content', '')
            
            metadata = {
                'ticker': ticker,
                'document_type': 'sec_filing',
                'filing_type': filing.get('type', ''),
                'date': filing.get('date', ''),
                'period': filing.get('period', ''),
                'source': 'sec_filings'
            }
            
            documents.append({
                'content': content,
                'metadata': metadata
            })
        
        return self.ingest_financial_documents(documents)
    
    def ingest_analyst_reports(self, 
                              ticker: str, 
                              reports_data: List[Dict[str, Any]]) -> List[str]:
        """
        Ingest analyst research reports.
        
        Args:
            ticker: Stock ticker symbol
            reports_data: List of analyst report data
            
        Returns:
            List of document IDs
        """
        documents = []
        for report in reports_data:
            content = f"Analyst Report for {ticker}\n"
            content += f"Firm: {report.get('firm', 'Unknown')}\n"
            content += f"Analyst: {report.get('analyst', 'Unknown')}\n"
            content += f"Date: {report.get('date', 'Unknown')}\n"
            content += f"Rating: {report.get('rating', 'Unknown')}\n"
            content += f"Price Target: {report.get('price_target', 'Unknown')}\n\n"
            content += report.get('content', '')
            
            metadata = {
                'ticker': ticker,
                'document_type': 'analyst_report',
                'firm': report.get('firm', ''),
                'analyst': report.get('analyst', ''),
                'date': report.get('date', ''),
                'rating': report.get('rating', ''),
                'price_target': report.get('price_target', ''),
                'source': 'analyst_reports'
            }
            
            documents.append({
                'content': content,
                'metadata': metadata
            })
        
        return self.ingest_financial_documents(documents)
    
    def ingest_news_articles(self, 
                            ticker: str, 
                            news_data: List[Dict[str, Any]]) -> List[str]:
        """
        Ingest news articles.
        
        Args:
            ticker: Stock ticker symbol
            news_data: List of news article data
            
        Returns:
            List of document IDs
        """
        documents = []
        for article in news_data:
            content = f"News Article: {article.get('headline', 'No headline')}\n"
            content += f"Source: {article.get('source', 'Unknown')}\n"
            content += f"Date: {article.get('date', 'Unknown')}\n"
            content += f"URL: {article.get('url', 'Unknown')}\n\n"
            content += article.get('content', article.get('summary', ''))
            
            metadata = {
                'ticker': ticker,
                'document_type': 'news_article',
                'headline': article.get('headline', ''),
                'source': article.get('source', ''),
                'date': article.get('date', ''),
                'url': article.get('url', ''),
                'sentiment': article.get('sentiment', 'neutral'),
                'source': 'news_articles'
            }
            
            documents.append({
                'content': content,
                'metadata': metadata
            })
        
        return self.ingest_financial_documents(documents)
    
    def ingest_market_data_analysis(self, 
                                   ticker: str, 
                                   market_data: Dict[str, Any]) -> str:
        """
        Ingest market data analysis.
        
        Args:
            ticker: Stock ticker symbol
            market_data: Market data analysis
            
        Returns:
            Document ID
        """
        content = f"Market Data Analysis for {ticker}\n"
        content += f"Date: {market_data.get('date', 'Unknown')}\n"
        content += f"Price: {market_data.get('price', 'Unknown')}\n"
        content += f"Volume: {market_data.get('volume', 'Unknown')}\n"
        content += f"Technical Indicators: {json.dumps(market_data.get('indicators', {}), indent=2)}\n\n"
        content += market_data.get('analysis', '')
        
        metadata = {
            'ticker': ticker,
            'document_type': 'market_analysis',
            'date': market_data.get('date', ''),
            'price': market_data.get('price', ''),
            'volume': market_data.get('volume', ''),
            'source': 'market_data'
        }
        
        return self.document_store.add_document(content, metadata)
    
    def query_knowledge_base(self, 
                            query: str,
                            context_type: str = "general",
                            ticker: Optional[str] = None,
                            trade_date: Optional[str] = None,
                            n_results: int = 5) -> Dict[str, Any]:
        """
        Query the knowledge base using RAG.
        
        Args:
            query: Search query
            context_type: Type of context
            ticker: Stock ticker symbol
            trade_date: Trading date
            n_results: Number of results to return
            
        Returns:
            RAG pipeline result
        """
        return self.pipeline.process_query(
            query=query,
            context_type=context_type,
            ticker=ticker,
            trade_date=trade_date,
            n_results=n_results
        )
    
    def enhance_agent_with_rag(self, 
                              agent_type: str,
                              current_analysis: str,
                              ticker: str,
                              trade_date: str) -> str:
        """
        Enhance agent analysis using RAG.
        
        Args:
            agent_type: Type of agent
            current_analysis: Current analysis
            ticker: Stock ticker
            trade_date: Trading date
            
        Returns:
            Enhanced analysis
        """
        return self.pipeline.enhance_agent_analysis(
            agent_type=agent_type,
            current_analysis=current_analysis,
            ticker=ticker,
            trade_date=trade_date
        )
    
    def get_knowledge_base_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge base."""
        return {
            "document_store_stats": self.document_store.get_collection_stats(),
            "pipeline_stats": self.pipeline.get_retrieval_stats()
        }
    
    def search_similar_situations(self, 
                                 situation: str, 
                                 ticker: Optional[str] = None,
                                 n_results: int = 3) -> List[Dict[str, Any]]:
        """
        Search for similar trading situations.
        
        Args:
            situation: Current trading situation
            ticker: Stock ticker symbol
            n_results: Number of results to return
            
        Returns:
            List of similar situations
        """
        return self.retriever.retrieve(
            query=situation,
            n_results=n_results,
            ticker=ticker,
            context_type="general"
        )
    
    def clear_knowledge_base(self) -> bool:
        """Clear all documents from the knowledge base."""
        return self.document_store.clear_collection()
