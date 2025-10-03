"""
Enhanced Trading Graph with RAG Integration

Extends the original TradingAgentsGraph with RAG capabilities.
"""

from typing import Dict, Any, Optional
from .integration import RAGTradingAgentsIntegration
from ..graph.trading_graph import TradingAgentsGraph


class RAGEnhancedTradingGraph(TradingAgentsGraph):
    """Trading graph enhanced with RAG capabilities."""
    
    def __init__(self, 
                 selected_analysts=["market", "social", "news", "fundamentals"],
                 debug=False,
                 config: Dict[str, Any] = None,
                 rag_enabled: bool = True):
        """
        Initialize enhanced trading graph with RAG.
        
        Args:
            selected_analysts: List of analyst types to include
            debug: Whether to run in debug mode
            config: Configuration dictionary
            rag_enabled: Whether to enable RAG functionality
        """
        # Initialize base trading graph
        super().__init__(selected_analysts, debug, config)
        
        # Initialize RAG integration
        self.rag_integration = RAGTradingAgentsIntegration(self.config)
        self.rag_enabled = rag_enabled
        
        if not rag_enabled:
            self.rag_integration.disable_rag()
    
    def propagate(self, company_name, trade_date):
        """
        Run the enhanced trading agents graph with RAG.
        
        Args:
            company_name: Company ticker symbol
            trade_date: Trading date
            
        Returns:
            Tuple of (final_state, processed_decision)
        """
        # Run base propagation
        final_state, decision = super().propagate(company_name, trade_date)
        
        # Enhance with RAG if enabled
        if self.rag_enabled:
            # Ingest session data for future reference
            self.rag_integration.ingest_trading_session_data(
                ticker=company_name,
                trade_date=trade_date,
                session_data=final_state
            )
        
        return final_state, decision
    
    def enhance_analyst_analysis(self, 
                                analyst_type: str,
                                current_analysis: str,
                                ticker: str,
                                trade_date: str,
                                additional_data: Optional[Dict[str, Any]] = None) -> str:
        """
        Enhance analyst analysis with RAG.
        
        Args:
            analyst_type: Type of analyst ('market', 'fundamentals', 'news', 'social')
            current_analysis: Current analysis from the agent
            ticker: Stock ticker symbol
            trade_date: Trading date
            additional_data: Additional data for ingestion
            
        Returns:
            Enhanced analysis
        """
        if not self.rag_enabled:
            return current_analysis
        
        if analyst_type == "market":
            return self.rag_integration.enhance_market_analyst(
                current_analysis, ticker, trade_date, additional_data
            )
        elif analyst_type == "fundamentals":
            return self.rag_integration.enhance_fundamentals_analyst(
                current_analysis, ticker, trade_date, additional_data
            )
        elif analyst_type == "news":
            return self.rag_integration.enhance_news_analyst(
                current_analysis, ticker, trade_date, additional_data
            )
        elif analyst_type == "social":
            return self.rag_integration.enhance_social_analyst(
                current_analysis, ticker, trade_date, additional_data
            )
        else:
            return current_analysis
    
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
        
        return self.rag_integration.enhance_trader_decision(
            current_decision, ticker, trade_date, market_context
        )
    
    def query_knowledge_base(self, 
                            query: str,
                            context_type: str = "general",
                            ticker: Optional[str] = None,
                            trade_date: Optional[str] = None,
                            n_results: int = 5) -> Dict[str, Any]:
        """
        Query the knowledge base directly.
        
        Args:
            query: Search query
            context_type: Type of context
            ticker: Stock ticker symbol
            trade_date: Trading date
            n_results: Number of results to return
            
        Returns:
            RAG pipeline result
        """
        if not self.rag_enabled:
            return {"error": "RAG is disabled"}
        
        return self.rag_integration.knowledge_base.query_knowledge_base(
            query=query,
            context_type=context_type,
            ticker=ticker,
            trade_date=trade_date,
            n_results=n_results
        )
    
    def ingest_documents(self, 
                        documents: list,
                        document_type: str = "general") -> List[str]:
        """
        Ingest documents into the knowledge base.
        
        Args:
            documents: List of documents to ingest
            document_type: Type of documents
            
        Returns:
            List of document IDs
        """
        if not self.rag_enabled:
            return []
        
        return self.rag_integration.knowledge_base.ingest_financial_documents(documents)
    
    def get_rag_stats(self) -> Dict[str, Any]:
        """Get RAG system statistics."""
        if not self.rag_enabled:
            return {"rag_enabled": False}
        
        stats = self.rag_integration.get_rag_stats()
        stats["rag_enabled"] = True
        return stats
    
    def enable_rag(self) -> None:
        """Enable RAG functionality."""
        self.rag_enabled = True
        self.rag_integration.enable_rag()
    
    def disable_rag(self) -> None:
        """Disable RAG functionality."""
        self.rag_enabled = False
        self.rag_integration.disable_rag()
