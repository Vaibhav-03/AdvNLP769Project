"""
RAG Integration Example for TradingAgents

This example demonstrates how to use the RAG-enhanced trading agents system.
"""

import os
import sys
from datetime import datetime, timedelta

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tradingagents.rag.enhanced_trading_graph import RAGEnhancedTradingGraph
from tradingagents.default_config import DEFAULT_CONFIG


def main():
    """Main example function."""
    
    # Configuration with RAG enabled
    config = DEFAULT_CONFIG.copy()
    config["rag_enabled"] = True
    
    # Initialize enhanced trading graph
    print("Initializing RAG-Enhanced Trading Graph...")
    trading_graph = RAGEnhancedTradingGraph(
        selected_analysts=["market", "social", "news", "fundamentals"],
        debug=True,
        config=config,
        rag_enabled=True
    )
    
    # Example 1: Basic trading decision with RAG
    print("\n" + "="*50)
    print("Example 1: Basic Trading Decision with RAG")
    print("="*50)
    
    ticker = "AAPL"
    trade_date = "2024-01-15"
    
    print(f"Running trading analysis for {ticker} on {trade_date}...")
    final_state, decision = trading_graph.propagate(ticker, trade_date)
    
    print(f"Final Decision: {decision}")
    print(f"RAG Stats: {trading_graph.get_rag_stats()}")
    
    # Example 2: Ingest sample documents
    print("\n" + "="*50)
    print("Example 2: Ingesting Sample Documents")
    print("="*50)
    
    sample_documents = [
        {
            'content': f"Apple Inc. reported strong Q4 earnings with revenue growth of 8% year-over-year. iPhone sales exceeded expectations, driven by strong demand for the iPhone 15 series. Services revenue continued to grow, reaching a new record high.",
            'metadata': {
                'ticker': 'AAPL',
                'document_type': 'earnings_report',
                'date': '2024-01-10',
                'quarter': 'Q4 2023',
                'source': 'earnings'
            }
        },
        {
            'content': f"Technical analysis shows AAPL breaking above key resistance at $180. RSI indicates overbought conditions but momentum remains strong. Volume is above average, supporting the breakout.",
            'metadata': {
                'ticker': 'AAPL',
                'document_type': 'technical_analysis',
                'date': '2024-01-12',
                'source': 'technical_analysis'
            }
        },
        {
            'content': f"Analyst upgrades for AAPL following strong earnings. Price target raised to $200 by multiple firms. Positive sentiment driven by AI integration in upcoming products.",
            'metadata': {
                'ticker': 'AAPL',
                'document_type': 'analyst_report',
                'date': '2024-01-14',
                'source': 'analyst_reports'
            }
        }
    ]
    
    document_ids = trading_graph.ingest_documents(sample_documents)
    print(f"Ingested {len(document_ids)} documents")
    
    # Example 3: Query knowledge base
    print("\n" + "="*50)
    print("Example 3: Querying Knowledge Base")
    print("="*50)
    
    queries = [
        "What are the recent earnings trends for AAPL?",
        "What is the technical analysis outlook for AAPL?",
        "What do analysts think about AAPL's future prospects?"
    ]
    
    for query in queries:
        print(f"\nQuery: {query}")
        result = trading_graph.query_knowledge_base(
            query=query,
            context_type="general",
            ticker="AAPL",
            trade_date=trade_date,
            n_results=2
        )
        
        print(f"Retrieved {len(result.get('retrieved_documents', []))} documents")
        print(f"Generated Response: {result.get('generated_response', 'No response')[:200]}...")
    
    # Example 4: Enhanced analysis
    print("\n" + "="*50)
    print("Example 4: Enhanced Analysis")
    print("="*50)
    
    # Simulate analyst analysis
    market_analysis = "AAPL shows strong technical indicators with RSI at 65 and MACD showing bullish divergence. Price is above 50-day moving average."
    
    enhanced_analysis = trading_graph.enhance_analyst_analysis(
        analyst_type="market",
        current_analysis=market_analysis,
        ticker="AAPL",
        trade_date=trade_date
    )
    
    print("Original Analysis:")
    print(market_analysis)
    print("\nEnhanced Analysis:")
    print(enhanced_analysis)
    
    # Example 5: RAG statistics
    print("\n" + "="*50)
    print("Example 5: RAG System Statistics")
    print("="*50)
    
    stats = trading_graph.get_rag_stats()
    print(f"RAG Enabled: {stats.get('rag_enabled', False)}")
    print(f"Total Documents: {stats.get('document_store_stats', {}).get('total_documents', 0)}")
    print(f"Collection Name: {stats.get('document_store_stats', {}).get('collection_name', 'N/A')}")
    
    print("\nRAG Integration Example completed successfully!")


if __name__ == "__main__":
    main()
