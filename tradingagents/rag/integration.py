"""
RAG Integration with TradingAgents

Integrates RAG framework with existing TradingAgents system.
Provides enhanced analysis capabilities for all agent types.
"""

from typing import Dict, Any, Optional, List
from datetime import datetime

from .knowledge_base import FinancialKnowledgeBase
from .rag_pipeline import RAGPipeline


class RAGTradingAgentsIntegration:
    """Integration layer between RAG framework and TradingAgents."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize RAG integration.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.knowledge_base = FinancialKnowledgeBase(config)
        self.rag_enabled = config.get("rag_enabled", True)
    
    def enhance_market_analyst(self, 
                              current_analysis: str,
                              ticker: str,
                              trade_date: str,
                              market_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Enhance market analyst with RAG.
        
        Args:
            current_analysis: Current market analysis
            ticker: Stock ticker symbol
            trade_date: Trading date
            market_data: Additional market data
            
        Returns:
            Enhanced market analysis
        """
        if not self.rag_enabled:
            return current_analysis
        
        # Ingest current market data if provided
        if market_data:
            self.knowledge_base.ingest_market_data_analysis(ticker, market_data)
        
        # Enhance analysis with RAG
        enhanced_analysis = self.knowledge_base.enhance_agent_with_rag(
            agent_type="market",
            current_analysis=current_analysis,
            ticker=ticker,
            trade_date=trade_date
        )
        
        return enhanced_analysis
    
    def enhance_fundamentals_analyst(self, 
                                    current_analysis: str,
                                    ticker: str,
                                    trade_date: str,
                                    fundamentals_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Enhance fundamentals analyst with RAG.
        
        Args:
            current_analysis: Current fundamentals analysis
            ticker: Stock ticker symbol
            trade_date: Trading date
            fundamentals_data: Additional fundamentals data
            
        Returns:
            Enhanced fundamentals analysis
        """
        if not self.rag_enabled:
            return current_analysis
        
        # Ingest fundamentals data if provided
        if fundamentals_data:
            self.knowledge_base.ingest_financial_documents([{
                'content': f"Fundamentals Analysis for {ticker}\n{fundamentals_data.get('content', '')}",
                'metadata': {
                    'ticker': ticker,
                    'document_type': 'fundamentals_analysis',
                    'date': trade_date,
                    'source': 'fundamentals_data'
                }
            }])
        
        # Enhance analysis with RAG
        enhanced_analysis = self.knowledge_base.enhance_agent_with_rag(
            agent_type="fundamentals",
            current_analysis=current_analysis,
            ticker=ticker,
            trade_date=trade_date
        )
        
        return enhanced_analysis
    
    def enhance_news_analyst(self, 
                            current_analysis: str,
                            ticker: str,
                            trade_date: str,
                            news_data: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Enhance news analyst with RAG.
        
        Args:
            current_analysis: Current news analysis
            ticker: Stock ticker symbol
            trade_date: Trading date
            news_data: Additional news data
            
        Returns:
            Enhanced news analysis
        """
        if not self.rag_enabled:
            return current_analysis
        
        # Ingest news data if provided
        if news_data:
            self.knowledge_base.ingest_news_articles(ticker, news_data)
        
        # Enhance analysis with RAG
        enhanced_analysis = self.knowledge_base.enhance_agent_with_rag(
            agent_type="news",
            current_analysis=current_analysis,
            ticker=ticker,
            trade_date=trade_date
        )
        
        return enhanced_analysis
    
    def enhance_social_analyst(self, 
                              current_analysis: str,
                              ticker: str,
                              trade_date: str,
                              social_data: Optional[List[Dict[str, Any]]] = None) -> str:
        """
        Enhance social media analyst with RAG.
        
        Args:
            current_analysis: Current social media analysis
            ticker: Stock ticker symbol
            trade_date: Trading date
            social_data: Additional social media data
            
        Returns:
            Enhanced social media analysis
        """
        if not self.rag_enabled:
            return current_analysis
        
        # Ingest social data if provided
        if social_data:
            documents = []
            for post in social_data:
                content = f"Social Media Post\nPlatform: {post.get('platform', 'Unknown')}\n"
                content += f"Author: {post.get('author', 'Unknown')}\n"
                content += f"Date: {post.get('date', 'Unknown')}\n"
                content += f"Content: {post.get('content', '')}\n"
                content += f"Sentiment: {post.get('sentiment', 'neutral')}\n"
                
                documents.append({
                    'content': content,
                    'metadata': {
                        'ticker': ticker,
                        'document_type': 'social_media',
                        'platform': post.get('platform', ''),
                        'author': post.get('author', ''),
                        'date': post.get('date', ''),
                        'sentiment': post.get('sentiment', 'neutral'),
                        'source': 'social_media'
                    }
                })
            
            self.knowledge_base.ingest_financial_documents(documents)
        
        # Enhance analysis with RAG
        enhanced_analysis = self.knowledge_base.enhance_agent_with_rag(
            agent_type="social",
            current_analysis=current_analysis,
            ticker=ticker,
            trade_date=trade_date
        )
        
        return enhanced_analysis
    
    def enhance_trader_decision(self, 
                               current_decision: str,
                               ticker: str,
                               trade_date: str,
                               market_context: Dict[str, str]) -> str:
        """
        Enhance trader decision with RAG.
        
        Args:
            current_decision: Current trading decision
            ticker: Stock ticker symbol
            trade_date: Trading date
            market_context: Context from all analysts
            
        Returns:
            Enhanced trading decision
        """
        if not self.rag_enabled:
            return current_decision
        
        # Create comprehensive context
        context_query = f"Trading decision for {ticker} on {trade_date}\n"
        context_query += f"Market Analysis: {market_context.get('market_report', '')}\n"
        context_query += f"News Analysis: {market_context.get('news_report', '')}\n"
        context_query += f"Fundamentals Analysis: {market_context.get('fundamentals_report', '')}\n"
        context_query += f"Social Analysis: {market_context.get('sentiment_report', '')}\n"
        context_query += f"Current Decision: {current_decision}"
        
        # Query knowledge base for similar situations
        similar_situations = self.knowledge_base.search_similar_situations(
            situation=context_query,
            ticker=ticker,
            n_results=3
        )
        
        # Generate enhanced decision
        if similar_situations:
            rag_result = self.knowledge_base.query_knowledge_base(
                query=context_query,
                context_type="general",
                ticker=ticker,
                trade_date=trade_date,
                n_results=3
            )
            
            enhanced_decision = f"""## Enhanced Trading Decision

### Original Decision:
{current_decision}

### Historical Context from Knowledge Base:
{rag_result['generated_response']}

### Final Recommendation:
Based on the analysis above and historical patterns, the trading decision should consider both current market conditions and similar historical situations.
"""
            return enhanced_decision
        
        return current_decision
    
    def ingest_trading_session_data(self, 
                                   ticker: str,
                                   trade_date: str,
                                   session_data: Dict[str, Any]) -> None:
        """
        Ingest trading session data for future reference.
        
        Args:
            ticker: Stock ticker symbol
            trade_date: Trading date
            session_data: Complete session data
        """
        if not self.rag_enabled:
            return
        
        # Create comprehensive session document
        content = f"Trading Session for {ticker} on {trade_date}\n"
        content += f"Final Decision: {session_data.get('final_trade_decision', 'Unknown')}\n"
        content += f"Market Report: {session_data.get('market_report', '')}\n"
        content += f"News Report: {session_data.get('news_report', '')}\n"
        content += f"Fundamentals Report: {session_data.get('fundamentals_report', '')}\n"
        content += f"Sentiment Report: {session_data.get('sentiment_report', '')}\n"
        content += f"Investment Plan: {session_data.get('investment_plan', '')}\n"
        
        metadata = {
            'ticker': ticker,
            'document_type': 'trading_session',
            'date': trade_date,
            'final_decision': session_data.get('final_trade_decision', ''),
            'source': 'trading_sessions'
        }
        
        self.knowledge_base.document_store.add_document(content, metadata)
    
    def get_rag_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        return self.knowledge_base.get_knowledge_base_stats()
    
    def enable_rag(self) -> None:
        """Enable RAG functionality."""
        self.rag_enabled = True
    
    def disable_rag(self) -> None:
        """Disable RAG functionality."""
        self.rag_enabled = False
