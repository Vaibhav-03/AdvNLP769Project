# RAG Integration Guide for TradingAgents

This guide provides comprehensive instructions for integrating the RAG (Retrieval-Augmented Generation) framework into your TradingAgents project.

## Overview

The RAG framework enhances TradingAgents by providing access to a comprehensive knowledge base of financial documents, enabling more informed trading decisions through retrieval-augmented analysis.

## Installation and Setup

### 1. Install Additional Dependencies

Add these dependencies to your `requirements.txt`:

```txt
# RAG Framework Dependencies
chromadb>=0.4.0
langchain-chroma>=0.1.0
sentence-transformers>=2.2.0
faiss-cpu>=1.7.4  # or faiss-gpu for GPU support
```

Install with:
```bash
pip install -r requirements.txt
```

### 2. Update Configuration

Update your configuration to enable RAG:

```python
from tradingagents.default_config import DEFAULT_CONFIG
from tradingagents.rag_config import update_config_with_rag

# Enable RAG in configuration
config = update_config_with_rag(DEFAULT_CONFIG)
config["rag_enabled"] = True
```

## Basic Integration

### 1. Replace TradingAgentsGraph with RAGEnhancedTradingGraph

```python
# Before (original)
from tradingagents.graph.trading_graph import TradingAgentsGraph

trading_graph = TradingAgentsGraph(
    selected_analysts=["market", "social", "news", "fundamentals"],
    config=config
)

# After (with RAG)
from tradingagents.rag.enhanced_trading_graph import RAGEnhancedTradingGraph

trading_graph = RAGEnhancedTradingGraph(
    selected_analysts=["market", "social", "news", "fundamentals"],
    config=config,
    rag_enabled=True
)
```

### 2. Run Trading Analysis

The interface remains the same:

```python
# Run trading analysis (now enhanced with RAG)
final_state, decision = trading_graph.propagate("AAPL", "2024-01-15")
print(f"Trading Decision: {decision}")
```

## Advanced Usage

### 1. Ingest Financial Documents

```python
# Ingest earnings call transcripts
earnings_data = [
    {
        'date': '2024-01-10',
        'quarter': 'Q4 2023',
        'transcript': 'Apple Inc. Q4 2023 Earnings Call Transcript...'
    }
]

document_ids = trading_graph.rag_integration.knowledge_base.ingest_earnings_calls(
    ticker="AAPL",
    earnings_data=earnings_data
)

# Ingest SEC filings
sec_filings = [
    {
        'type': '10-K',
        'date': '2023-12-31',
        'period': 'FY 2023',
        'content': 'Annual Report on Form 10-K...'
    }
]

document_ids = trading_graph.rag_integration.knowledge_base.ingest_sec_filings(
    ticker="AAPL",
    filings_data=sec_filings
)
```

### 2. Query Knowledge Base

```python
# Query for specific information
result = trading_graph.query_knowledge_base(
    query="What are Apple's recent revenue trends?",
    context_type="fundamentals",
    ticker="AAPL",
    trade_date="2024-01-15",
    n_results=3
)

print("Retrieved Documents:", len(result['retrieved_documents']))
print("Generated Response:", result['generated_response'])
```

### 3. Enhance Individual Analyst Analysis

```python
# Enhance market analyst with RAG
market_analysis = "AAPL shows strong technical indicators..."
enhanced_analysis = trading_graph.enhance_analyst_analysis(
    analyst_type="market",
    current_analysis=market_analysis,
    ticker="AAPL",
    trade_date="2024-01-15"
)
```

## Configuration Options

### 1. RAG System Settings

```python
config = {
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

### 2. Data Source Configuration

```python
config["data_sources"] = {
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

## Integration with Existing Workflows

### 1. CLI Integration

Update your CLI to use the enhanced graph:

```python
# In cli/main.py
from tradingagents.rag.enhanced_trading_graph import RAGEnhancedTradingGraph

# Replace TradingAgentsGraph with RAGEnhancedTradingGraph
trading_graph = RAGEnhancedTradingGraph(
    selected_analysts=selected_analysts,
    debug=debug,
    config=config,
    rag_enabled=True
)
```

### 2. Evaluation Framework Integration

Update evaluation scripts:

```python
# In evaluation/evaluator.py
from tradingagents.rag.enhanced_trading_graph import RAGEnhancedTradingGraph

# Use RAG-enhanced graph for evaluation
trading_graph = RAGEnhancedTradingGraph(
    selected_analysts=analysts,
    config=config,
    rag_enabled=True
)
```

### 3. Custom Agent Integration

For custom agents, use the enhancement methods:

```python
# Enhance custom analysis
enhanced_analysis = trading_graph.enhance_analyst_analysis(
    analyst_type="custom",
    current_analysis=your_analysis,
    ticker=ticker,
    trade_date=trade_date
)
```

## Data Ingestion Strategies

### 1. Batch Document Ingestion

```python
# Prepare documents for batch ingestion
documents = []
for doc in your_documents:
    documents.append({
        'content': doc['text'],
        'metadata': {
            'ticker': doc['ticker'],
            'document_type': doc['type'],
            'date': doc['date'],
            'source': doc['source']
        }
    })

# Ingest in batch
document_ids = trading_graph.ingest_documents(documents)
```

### 2. Real-time Data Ingestion

```python
# Ingest data as it becomes available
def ingest_realtime_data(ticker, data_type, content, metadata):
    document = {
        'content': content,
        'metadata': {
            'ticker': ticker,
            'document_type': data_type,
            'date': datetime.now().strftime('%Y-%m-%d'),
            **metadata
        }
    }
    
    trading_graph.ingest_documents([document])
```

## Performance Optimization

### 1. Caching

Enable caching for better performance:

```python
config["enable_caching"] = True
config["cache_ttl"] = 3600  # 1 hour
```

### 2. Batch Processing

Use batch processing for large datasets:

```python
config["batch_size"] = 10
config["max_concurrent_requests"] = 5
```

### 3. Memory Management

Optimize memory usage:

```python
config["max_document_length"] = 4000
config["chunk_size"] = 1000
config["chunk_overlap"] = 200
```

## Monitoring and Debugging

### 1. Enable Debug Mode

```python
trading_graph = RAGEnhancedTradingGraph(
    debug=True,
    rag_enabled=True
)
```

### 2. Monitor RAG Statistics

```python
# Get RAG system statistics
stats = trading_graph.get_rag_stats()
print(f"Total documents: {stats['document_store_stats']['total_documents']}")
print(f"RAG enabled: {stats['rag_enabled']}")
```

### 3. Log RAG Operations

```python
config["log_rag_queries"] = True
config["log_retrieval_stats"] = True
config["log_generation_stats"] = True
```

## Troubleshooting

### Common Issues

1. **RAG Not Working**
   - Check if `rag_enabled=True` in configuration
   - Verify ChromaDB installation
   - Check embedding model compatibility

2. **Low Retrieval Quality**
   - Adjust `min_similarity_threshold`
   - Check document quality and metadata
   - Try different retrieval strategies

3. **Performance Issues**
   - Enable caching
   - Reduce `max_retrieval_results`
   - Optimize batch sizes

4. **Memory Issues**
   - Reduce `max_document_length`
   - Implement document chunking
   - Monitor embedding storage

### Debug Commands

```python
# Check RAG status
print(f"RAG enabled: {trading_graph.rag_enabled}")

# Get collection statistics
stats = trading_graph.get_rag_stats()
print(f"Collection stats: {stats}")

# Test retrieval
result = trading_graph.query_knowledge_base(
    query="test query",
    n_results=1
)
print(f"Retrieval test: {len(result['retrieved_documents'])} documents")
```

## Migration from Original System

### 1. Gradual Migration

Start with RAG disabled to ensure compatibility:

```python
trading_graph = RAGEnhancedTradingGraph(
    rag_enabled=False  # Start with RAG disabled
)

# Test existing functionality
final_state, decision = trading_graph.propagate("AAPL", "2024-01-15")

# Enable RAG when ready
trading_graph.enable_rag()
```

### 2. A/B Testing

Compare performance with and without RAG:

```python
# Test without RAG
trading_graph.disable_rag()
result_without_rag = trading_graph.propagate("AAPL", "2024-01-15")

# Test with RAG
trading_graph.enable_rag()
result_with_rag = trading_graph.propagate("AAPL", "2024-01-15")

# Compare results
print("Without RAG:", result_without_rag[1])
print("With RAG:", result_with_rag[1])
```

## Best Practices

### 1. Document Quality
- Ensure high-quality source documents
- Use consistent metadata schemas
- Regular data validation and cleanup

### 2. Query Optimization
- Use specific, focused queries
- Leverage metadata filtering
- Monitor retrieval performance

### 3. System Maintenance
- Regular knowledge base updates
- Performance monitoring
- Error handling and recovery

### 4. Security
- Secure document storage
- Access control for sensitive data
- Audit logging for compliance

## Example Implementation

See `examples/rag_integration_example.py` for a complete working example that demonstrates:

- Basic RAG integration
- Document ingestion
- Knowledge base querying
- Enhanced analysis generation
- System monitoring

## Support

For issues and questions:

1. Check the troubleshooting section
2. Review the RAG module documentation
3. Examine the example implementations
4. Monitor system logs and statistics

## Future Enhancements

Planned features include:

- Multi-modal document support
- Real-time document ingestion
- Advanced query expansion
- Custom embedding models
- Federated knowledge bases

The RAG framework is designed to be extensible and can be customized for specific use cases and requirements.
