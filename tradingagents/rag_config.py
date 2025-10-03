"""
RAG Configuration for TradingAgents

Configuration settings for RAG integration with TradingAgents.
"""

import os

# RAG-specific configuration
RAG_CONFIG = {
    # RAG System Settings
    "rag_enabled": True,
    "rag_collection_name": "financial_knowledge_base",
    
    # Retrieval Settings
    "retrieval_strategy": "hybrid",  # "semantic", "keyword", or "hybrid"
    "semantic_weight": 0.7,
    "keyword_weight": 0.3,
    "min_similarity_threshold": 0.6,
    "max_retrieval_results": 5,
    
    # Document Processing
    "max_document_length": 4000,
    "chunk_size": 1000,
    "chunk_overlap": 200,
    
    # Knowledge Base Settings
    "auto_ingest_session_data": True,
    "ingest_earnings_calls": True,
    "ingest_sec_filings": True,
    "ingest_analyst_reports": True,
    "ingest_news_articles": True,
    "ingest_social_media": True,
    
    # Enhancement Settings
    "enhance_market_analyst": True,
    "enhance_fundamentals_analyst": True,
    "enhance_news_analyst": True,
    "enhance_social_analyst": True,
    "enhance_trader_decision": True,
    
    # Data Sources
    "data_sources": {
        "earnings_calls": {
            "enabled": True,
            "lookback_days": 365,
            "sources": ["seeking_alpha", "yahoo_finance"]
        },
        "sec_filings": {
            "enabled": True,
            "lookback_days": 365,
            "filing_types": ["10-K", "10-Q", "8-K"]
        },
        "analyst_reports": {
            "enabled": True,
            "lookback_days": 180,
            "sources": ["bloomberg", "reuters", "morningstar"]
        },
        "news_articles": {
            "enabled": True,
            "lookback_days": 30,
            "sources": ["reuters", "bloomberg", "cnbc", "wsj"]
        },
        "social_media": {
            "enabled": True,
            "lookback_days": 7,
            "platforms": ["twitter", "reddit", "stocktwits"]
        }
    },
    
    # Performance Settings
    "enable_caching": True,
    "cache_ttl": 3600,  # 1 hour
    "batch_size": 10,
    "max_concurrent_requests": 5,
    
    # Logging and Monitoring
    "log_rag_queries": True,
    "log_retrieval_stats": True,
    "log_generation_stats": True,
    "rag_log_level": "INFO"
}


def get_rag_config() -> dict:
    """Get RAG configuration."""
    return RAG_CONFIG.copy()


def update_config_with_rag(base_config: dict) -> dict:
    """Update base configuration with RAG settings."""
    updated_config = base_config.copy()
    updated_config.update(RAG_CONFIG)
    return updated_config
