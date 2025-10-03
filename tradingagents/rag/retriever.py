"""
RAG Retriever for TradingAgents

Implements various retrieval strategies for finding relevant financial information.
Supports hybrid search combining semantic and keyword search.
"""

from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
import re
from datetime import datetime, timedelta
from .document_store import DocumentStore


class BaseRetriever(ABC):
    """Abstract base class for retrievers."""
    
    @abstractmethod
    def retrieve(self, query: str, n_results: int = 5, **kwargs) -> List[Dict[str, Any]]:
        """Retrieve relevant documents for a query."""
        pass


class SemanticRetriever(BaseRetriever):
    """Semantic retriever using embeddings."""
    
    def __init__(self, document_store: DocumentStore):
        self.document_store = document_store
    
    def retrieve(self, 
                query: str, 
                n_results: int = 5,
                ticker: Optional[str] = None,
                document_type: Optional[str] = None,
                date_range: Optional[Tuple[str, str]] = None,
                min_similarity: float = 0.7) -> List[Dict[str, Any]]:
        """
        Retrieve documents using semantic similarity.
        
        Args:
            query: Search query
            n_results: Number of results to return
            ticker: Filter by ticker symbol
            document_type: Filter by document type
            date_range: Filter by date range (start_date, end_date)
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of relevant documents
        """
        results = self.document_store.search_documents(
            query=query,
            n_results=n_results,
            ticker=ticker,
            document_type=document_type,
            date_range=date_range
        )
        
        # Filter by similarity threshold
        filtered_results = [
            doc for doc in results 
            if doc['similarity_score'] >= min_similarity
        ]
        
        return filtered_results


class KeywordRetriever(BaseRetriever):
    """Keyword-based retriever using text matching."""
    
    def __init__(self, document_store: DocumentStore):
        self.document_store = document_store
    
    def retrieve(self, 
                query: str, 
                n_results: int = 5,
                ticker: Optional[str] = None,
                document_type: Optional[str] = None,
                date_range: Optional[Tuple[str, str]] = None) -> List[Dict[str, Any]]:
        """
        Retrieve documents using keyword matching.
        
        Args:
            query: Search query with keywords
            n_results: Number of results to return
            ticker: Filter by ticker symbol
            document_type: Filter by document type
            date_range: Filter by date range
            
        Returns:
            List of relevant documents
        """
        # Extract keywords from query
        keywords = self._extract_keywords(query)
        
        # Get all documents (we'll filter by keywords manually)
        all_docs = self.document_store.search_documents(
            query=query,
            n_results=100,  # Get more to filter
            ticker=ticker,
            document_type=document_type,
            date_range=date_range
        )
        
        # Score documents based on keyword matches
        scored_docs = []
        for doc in all_docs:
            score = self._calculate_keyword_score(doc['content'], keywords)
            if score > 0:
                doc['keyword_score'] = score
                scored_docs.append(doc)
        
        # Sort by keyword score and return top results
        scored_docs.sort(key=lambda x: x['keyword_score'], reverse=True)
        return scored_docs[:n_results]
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query."""
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
        words = re.findall(r'\b\w+\b', query.lower())
        keywords = [word for word in words if word not in stop_words and len(word) > 2]
        return keywords
    
    def _calculate_keyword_score(self, content: str, keywords: List[str]) -> float:
        """Calculate keyword match score for content."""
        content_lower = content.lower()
        score = 0
        for keyword in keywords:
            # Count occurrences
            count = content_lower.count(keyword)
            score += count
        
        # Normalize by content length
        return score / max(len(content.split()), 1)


class HybridRetriever(BaseRetriever):
    """Hybrid retriever combining semantic and keyword search."""
    
    def __init__(self, document_store: DocumentStore, 
                 semantic_weight: float = 0.7, 
                 keyword_weight: float = 0.3):
        self.semantic_retriever = SemanticRetriever(document_store)
        self.keyword_retriever = KeywordRetriever(document_store)
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight
    
    def retrieve(self, 
                query: str, 
                n_results: int = 5,
                ticker: Optional[str] = None,
                document_type: Optional[str] = None,
                date_range: Optional[Tuple[str, str]] = None,
                min_similarity: float = 0.6) -> List[Dict[str, Any]]:
        """
        Retrieve documents using hybrid approach.
        
        Args:
            query: Search query
            n_results: Number of results to return
            ticker: Filter by ticker symbol
            document_type: Filter by document type
            date_range: Filter by date range
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of relevant documents with combined scores
        """
        # Get semantic results
        semantic_results = self.semantic_retriever.retrieve(
            query=query,
            n_results=n_results * 2,  # Get more for combination
            ticker=ticker,
            document_type=document_type,
            date_range=date_range,
            min_similarity=min_similarity
        )
        
        # Get keyword results
        keyword_results = self.keyword_retriever.retrieve(
            query=query,
            n_results=n_results * 2,
            ticker=ticker,
            document_type=document_type,
            date_range=date_range
        )
        
        # Combine results
        combined_results = self._combine_results(semantic_results, keyword_results)
        
        # Sort by combined score and return top results
        combined_results.sort(key=lambda x: x['combined_score'], reverse=True)
        return combined_results[:n_results]
    
    def _combine_results(self, 
                        semantic_results: List[Dict[str, Any]], 
                        keyword_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Combine semantic and keyword results."""
        # Create a map of document IDs to results
        doc_map = {}
        
        # Add semantic results
        for doc in semantic_results:
            doc_id = doc['document_id']
            doc_map[doc_id] = {
                'content': doc['content'],
                'metadata': doc['metadata'],
                'document_id': doc_id,
                'semantic_score': doc['similarity_score'],
                'keyword_score': 0.0
            }
        
        # Add/update with keyword results
        for doc in keyword_results:
            doc_id = doc['document_id']
            if doc_id in doc_map:
                doc_map[doc_id]['keyword_score'] = doc.get('keyword_score', 0.0)
            else:
                doc_map[doc_id] = {
                    'content': doc['content'],
                    'metadata': doc['metadata'],
                    'document_id': doc_id,
                    'semantic_score': 0.0,
                    'keyword_score': doc.get('keyword_score', 0.0)
                }
        
        # Calculate combined scores
        for doc in doc_map.values():
            doc['combined_score'] = (
                self.semantic_weight * doc['semantic_score'] + 
                self.keyword_weight * doc['keyword_score']
            )
        
        return list(doc_map.values())


class RAGRetriever:
    """Main RAG retriever that orchestrates different retrieval strategies."""
    
    def __init__(self, document_store: DocumentStore, 
                 retrieval_strategy: str = "hybrid"):
        """
        Initialize RAG retriever.
        
        Args:
            document_store: Document store instance
            retrieval_strategy: Strategy to use ('semantic', 'keyword', 'hybrid')
        """
        self.document_store = document_store
        
        if retrieval_strategy == "semantic":
            self.retriever = SemanticRetriever(document_store)
        elif retrieval_strategy == "keyword":
            self.retriever = KeywordRetriever(document_store)
        elif retrieval_strategy == "hybrid":
            self.retriever = HybridRetriever(document_store)
        else:
            raise ValueError(f"Unknown retrieval strategy: {retrieval_strategy}")
    
    def retrieve(self, 
                query: str, 
                n_results: int = 5,
                ticker: Optional[str] = None,
                document_type: Optional[str] = None,
                date_range: Optional[Tuple[str, str]] = None,
                context_type: str = "general") -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for trading context.
        
        Args:
            query: Search query
            n_results: Number of results to return
            ticker: Filter by ticker symbol
            document_type: Filter by document type
            date_range: Filter by date range
            context_type: Type of context ('market', 'fundamentals', 'news', 'social')
            
        Returns:
            List of relevant documents
        """
        # Adjust retrieval parameters based on context type
        if context_type == "market":
            # Focus on technical analysis and market data
            document_type = document_type or "market_data"
            min_similarity = 0.7
        elif context_type == "fundamentals":
            # Focus on financial statements and company data
            document_type = document_type or "financial_statement"
            min_similarity = 0.8
        elif context_type == "news":
            # Focus on news and sentiment
            document_type = document_type or "news"
            min_similarity = 0.6
        elif context_type == "social":
            # Focus on social media and sentiment
            document_type = document_type or "social_media"
            min_similarity = 0.5
        else:
            min_similarity = 0.6
        
        # Retrieve documents
        results = self.retriever.retrieve(
            query=query,
            n_results=n_results,
            ticker=ticker,
            document_type=document_type,
            date_range=date_range,
            min_similarity=min_similarity
        )
        
        return results
