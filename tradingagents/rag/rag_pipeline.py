"""
RAG Pipeline for TradingAgents

Orchestrates the complete RAG workflow: retrieval, augmentation, and generation.
Integrates with existing trading agent workflows.
"""

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import json

from .document_store import DocumentStore
from .retriever import RAGRetriever
from .generator import RAGGenerator


class RAGPipeline:
    """Main RAG pipeline that orchestrates retrieval and generation."""
    
    def __init__(self, 
                 document_store: DocumentStore,
                 retriever: RAGRetriever,
                 generator: RAGGenerator,
                 config: Dict[str, Any]):
        """
        Initialize RAG pipeline.
        
        Args:
            document_store: Document store instance
            retriever: RAG retriever instance
            generator: RAG generator instance
            config: Configuration dictionary
        """
        self.document_store = document_store
        self.retriever = retriever
        self.generator = generator
        self.config = config
    
    def process_query(self, 
                     query: str,
                     context_type: str = "general",
                     ticker: Optional[str] = None,
                     trade_date: Optional[str] = None,
                     n_results: int = 5,
                     document_type: Optional[str] = None,
                     date_range: Optional[Tuple[str, str]] = None) -> Dict[str, Any]:
        """
        Process a query through the complete RAG pipeline.
        
        Args:
            query: Input query
            context_type: Type of context ('market', 'fundamentals', 'news', 'social')
            ticker: Stock ticker symbol
            trade_date: Trading date
            n_results: Number of documents to retrieve
            document_type: Filter by document type
            date_range: Filter by date range
            
        Returns:
            Dictionary containing query, retrieved docs, and generated response
        """
        # Step 1: Retrieve relevant documents
        retrieved_docs = self.retriever.retrieve(
            query=query,
            n_results=n_results,
            ticker=ticker,
            document_type=document_type,
            date_range=date_range,
            context_type=context_type
        )
        
        # Step 2: Generate response using retrieved documents
        generated_response = self.generator.generate(
            query=query,
            retrieved_docs=retrieved_docs,
            context_type=context_type,
            ticker=ticker,
            trade_date=trade_date
        )
        
        return {
            "query": query,
            "retrieved_documents": retrieved_docs,
            "generated_response": generated_response,
            "context_type": context_type,
            "ticker": ticker,
            "trade_date": trade_date,
            "timestamp": datetime.now().isoformat()
        }
    
    def enhance_agent_analysis(self, 
                              agent_type: str,
                              current_analysis: str,
                              ticker: str,
                              trade_date: str,
                              additional_context: Optional[Dict[str, Any]] = None) -> str:
        """
        Enhance agent analysis using RAG.
        
        Args:
            agent_type: Type of agent ('market', 'fundamentals', 'news', 'social')
            current_analysis: Current analysis from the agent
            ticker: Stock ticker symbol
            trade_date: Trading date
            additional_context: Additional context for retrieval
            
        Returns:
            Enhanced analysis
        """
        # Create query based on agent type and current analysis
        query = self._create_agent_query(agent_type, current_analysis, ticker)
        
        # Process through RAG pipeline
        result = self.process_query(
            query=query,
            context_type=agent_type,
            ticker=ticker,
            trade_date=trade_date,
            n_results=3  # Fewer docs for agent enhancement
        )
        
        # Combine current analysis with RAG-enhanced insights
        enhanced_analysis = self._combine_analysis(
            current_analysis, 
            result["generated_response"],
            agent_type
        )
        
        return enhanced_analysis
    
    def _create_agent_query(self, agent_type: str, analysis: str, ticker: str) -> str:
        """Create query for agent enhancement."""
        if agent_type == "market":
            return f"Technical analysis and market trends for {ticker}: {analysis[:200]}..."
        elif agent_type == "fundamentals":
            return f"Fundamental analysis and financial metrics for {ticker}: {analysis[:200]}..."
        elif agent_type == "news":
            return f"Recent news and sentiment analysis for {ticker}: {analysis[:200]}..."
        elif agent_type == "social":
            return f"Social media sentiment and public opinion for {ticker}: {analysis[:200]}..."
        else:
            return f"Analysis for {ticker}: {analysis[:200]}..."
    
    def _combine_analysis(self, 
                         current_analysis: str, 
                         rag_insights: str, 
                         agent_type: str) -> str:
        """Combine current analysis with RAG insights."""
        if agent_type == "market":
            return f"""## Market Analysis

### Current Analysis:
{current_analysis}

### Enhanced Insights from Knowledge Base:
{rag_insights}

### Combined Assessment:
The above analysis combines real-time market data with historical patterns and expert insights from our knowledge base to provide a comprehensive market assessment.
"""
        elif agent_type == "fundamentals":
            return f"""## Fundamental Analysis

### Current Analysis:
{current_analysis}

### Enhanced Insights from Knowledge Base:
{rag_insights}

### Combined Assessment:
This analysis integrates current financial data with historical trends and industry benchmarks to provide a thorough fundamental evaluation.
"""
        elif agent_type == "news":
            return f"""## News Analysis

### Current Analysis:
{current_analysis}

### Enhanced Insights from Knowledge Base:
{rag_insights}

### Combined Assessment:
The analysis above combines recent news developments with historical context and expert commentary to assess news impact on trading decisions.
"""
        elif agent_type == "social":
            return f"""## Social Media Analysis

### Current Analysis:
{current_analysis}

### Enhanced Insights from Knowledge Base:
{rag_insights}

### Combined Assessment:
This analysis merges current social sentiment with historical patterns and expert insights to evaluate public opinion trends.
"""
        else:
            return f"""## Analysis

### Current Analysis:
{current_analysis}

### Enhanced Insights from Knowledge Base:
{rag_insights}

### Combined Assessment:
The above analysis combines current data with historical context and expert insights for comprehensive evaluation.
"""
    
    def batch_process_queries(self, 
                             queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Process multiple queries in batch.
        
        Args:
            queries: List of query dictionaries with keys: query, context_type, ticker, etc.
            
        Returns:
            List of results for each query
        """
        results = []
        for query_dict in queries:
            result = self.process_query(**query_dict)
            results.append(result)
        return results
    
    def get_retrieval_stats(self) -> Dict[str, Any]:
        """Get statistics about retrieval performance."""
        collection_stats = self.document_store.get_collection_stats()
        return {
            "document_store_stats": collection_stats,
            "retrieval_strategy": self.retriever.retriever.__class__.__name__,
            "generation_strategy": "trading"
        }
