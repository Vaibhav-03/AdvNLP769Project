# RAG Framework for TradingAgents

This module provides Retrieval-Augmented Generation (RAG) capabilities for the TradingAgents system, enhancing trading decisions with relevant information from a comprehensive knowledge base.

## Overview

The RAG framework consists of several key components:

- **Document Store**: Manages storage and retrieval of financial documents
- **Retriever**: Implements various retrieval strategies (semantic, keyword, hybrid)
- **Generator**: Generates enhanced responses using retrieved documents
- **Knowledge Base**: Orchestrates document ingestion and management
- **Integration**: Seamlessly integrates with existing TradingAgents workflows

## Key Features

### 1. Multi-Modal Document Support
- Earnings call transcripts
- SEC filings (10-K, 10-Q, 8-K)
- Analyst research reports
- News articles and press releases
- Social media sentiment data
- Technical analysis reports
- Market data and indicators

### 2. Advanced Retrieval Strategies
- **Semantic Retrieval**: Uses embeddings for semantic similarity
- **Keyword Retrieval**: Traditional keyword-based search
- **Hybrid Retrieval**: Combines semantic and keyword approaches

### 3. Context-Aware Generation
- Specialized prompts for different analysis types
- Integration with existing agent workflows
- Historical pattern recognition
- Risk-aware decision making

### 4. Seamless Integration
- Drop-in replacement for existing TradingAgentsGraph
- Backward compatibility
- Configurable RAG enable/disable
- Performance monitoring

## Quick Start

### Basic Usage

```python
from tradingagents.rag.enhanced_trading_graph import RAGEnhancedTradingGraph
from tradingagents.default_config import DEFAULT_CONFIG

# Enable RAG in configuration
config = DEFAULT_CONFIG.copy()
config["rag_enabled"] = True

# Initialize enhanced trading graph
trading_graph = RAGEnhancedTradingGraph(
    selected_analysts=["market", "social", "news", "fundamentals"],
    config=config,
    rag_enabled=True
)

# Run trading analysis with RAG enhancement
final_state, decision = trading_graph.propagate("AAPL", "2024-01-15")
```

### Ingesting Documents

```python
# Ingest sample documents
documents = [
    {
        'content': 'Apple Inc. reported strong Q4 earnings...',
        'metadata': {
            'ticker': 'AAPL',
            'document_type': 'earnings_report',
            'date': '2024-01-10',
            'source': 'earnings'
        }
    }
]

document_ids = trading_graph.ingest_documents(documents)
```

### Querying Knowledge Base

```python
# Query the knowledge base
result = trading_graph.query_knowledge_base(
    query="What are the recent earnings trends for AAPL?",
    context_type="fundamentals",
    ticker="AAPL",
    trade_date="2024-01-15",
    n_results=3
)

print(result['generated_response'])
```

## Architecture

### Document Store
- Built on ChromaDB for efficient vector storage
- Supports metadata filtering and date ranges
- Automatic embedding generation
- Batch document ingestion

### Retrieval System
- Multiple retrieval strategies
- Configurable similarity thresholds
- Context-aware filtering
- Performance optimization

### Generation System
- Specialized prompts for trading contexts
- Integration with existing LLM infrastructure
- Response quality monitoring
- Error handling and fallbacks

### Knowledge Base
- Centralized document management
- Automatic data ingestion from various sources
- Document type classification
- Metadata enrichment

## Configuration

### RAG Settings

```python
RAG_CONFIG = {
    "rag_enabled": True,
    "retrieval_strategy": "hybrid",  # "semantic", "keyword", or "hybrid"
    "min_similarity_threshold": 0.6,
    "max_retrieval_results": 5,
    "enhance_market_analyst": True,
    "enhance_fundamentals_analyst": True,
    "enhance_news_analyst": True,
    "enhance_social_analyst": True,
    "enhance_trader_decision": True
}
```

### Data Sources

```python
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
    "news_articles": {
        "enabled": True,
        "lookback_days": 30,
        "sources": ["reuters", "bloomberg", "cnbc"]
    }
}
```

## Integration Points

### 1. Market Analyst Enhancement
- Technical analysis with historical patterns
- Market trend identification
- Indicator correlation analysis

### 2. Fundamentals Analyst Enhancement
- Financial statement analysis
- Industry benchmarking
- Valuation model integration

### 3. News Analyst Enhancement
- Sentiment analysis with historical context
- News impact assessment
- Event correlation analysis

### 4. Social Media Analyst Enhancement
- Sentiment trend analysis
- Influencer opinion tracking
- Community sentiment patterns

### 5. Trader Decision Enhancement
- Historical decision patterns
- Risk assessment with context
- Market condition correlation

## Performance Considerations

### Caching
- Document embeddings are cached
- Query results can be cached
- Configurable TTL settings

### Batch Processing
- Batch document ingestion
- Parallel retrieval processing
- Async generation when possible

### Memory Management
- Document chunking for large files
- Embedding compression
- Garbage collection optimization

## Monitoring and Logging

### Metrics
- Retrieval accuracy
- Generation quality
- Response time
- Document ingestion rate

### Logging
- Query logging
- Retrieval statistics
- Generation statistics
- Error tracking

## Best Practices

### 1. Document Quality
- Ensure high-quality source documents
- Proper metadata tagging
- Regular data validation

### 2. Query Optimization
- Use specific, focused queries
- Leverage metadata filtering
- Monitor retrieval performance

### 3. Response Quality
- Validate generated responses
- Monitor for hallucinations
- Implement quality checks

### 4. System Maintenance
- Regular knowledge base updates
- Performance monitoring
- Error handling and recovery

## Troubleshooting

### Common Issues

1. **Low Retrieval Quality**
   - Check document quality and metadata
   - Adjust similarity thresholds
   - Verify embedding model compatibility

2. **Slow Performance**
   - Enable caching
   - Optimize batch sizes
   - Check database performance

3. **Memory Issues**
   - Implement document chunking
   - Monitor embedding storage
   - Optimize batch processing

### Debug Mode

```python
# Enable debug mode for detailed logging
trading_graph = RAGEnhancedTradingGraph(
    debug=True,
    rag_enabled=True
)

# Check RAG statistics
stats = trading_graph.get_rag_stats()
print(f"Total documents: {stats['document_store_stats']['total_documents']}")
```

## Future Enhancements

### Planned Features
- Multi-modal document support (images, charts)
- Real-time document ingestion
- Advanced query expansion
- Custom embedding models
- Federated knowledge bases

### Research Areas
- Few-shot learning for document classification
- Adversarial retrieval robustness
- Multi-lingual document support
- Temporal reasoning in financial data

## Contributing

To contribute to the RAG framework:

1. Follow the existing code structure
2. Add comprehensive tests
3. Update documentation
4. Ensure backward compatibility
5. Monitor performance impact

## License

This RAG framework is part of the TradingAgents project and follows the same license terms.
