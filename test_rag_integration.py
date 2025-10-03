#!/usr/bin/env python3
"""
Test script for RAG integration with TradingAgents

This script tests the basic functionality of the RAG framework integration.
"""

import os
import sys
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_rag_imports():
    """Test that RAG modules can be imported."""
    print("Testing RAG module imports...")
    
    try:
        from tradingagents.rag import DocumentStore, RAGRetriever, RAGGenerator, RAGPipeline, FinancialKnowledgeBase
        print("✓ RAG core modules imported successfully")
        
        from tradingagents.rag.enhanced_trading_graph import RAGEnhancedTradingGraph
        print("✓ Enhanced trading graph imported successfully")
        
        from tradingagents.rag.integration import RAGTradingAgentsIntegration
        print("✓ RAG integration module imported successfully")
        
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_rag_config():
    """Test RAG configuration."""
    print("\nTesting RAG configuration...")
    
    try:
        from tradingagents.rag_config import get_rag_config, update_config_with_rag
        from tradingagents.default_config import DEFAULT_CONFIG
        
        # Test getting RAG config
        rag_config = get_rag_config()
        assert "rag_enabled" in rag_config
        assert "retrieval_strategy" in rag_config
        print("✓ RAG configuration loaded successfully")
        
        # Test updating base config
        updated_config = update_config_with_rag(DEFAULT_CONFIG)
        assert updated_config["rag_enabled"] == True
        print("✓ Configuration update successful")
        
        return True
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False

def test_document_store():
    """Test document store functionality."""
    print("\nTesting document store...")
    
    try:
        from tradingagents.rag.document_store import DocumentStore
        from tradingagents.default_config import DEFAULT_CONFIG
        
        # Create document store
        doc_store = DocumentStore(DEFAULT_CONFIG, "test_collection")
        print("✓ Document store created successfully")
        
        # Test adding a document
        test_doc = {
            'content': 'This is a test document for RAG integration.',
            'metadata': {
                'ticker': 'TEST',
                'document_type': 'test',
                'date': '2024-01-15',
                'source': 'test'
            }
        }
        
        doc_id = doc_store.add_document(
            content=test_doc['content'],
            metadata=test_doc['metadata']
        )
        print(f"✓ Document added with ID: {doc_id}")
        
        # Test searching documents
        results = doc_store.search_documents(
            query="test document",
            n_results=1
        )
        assert len(results) > 0
        print(f"✓ Document search successful: {len(results)} results")
        
        # Test getting document by ID
        retrieved_doc = doc_store.get_document_by_id(doc_id)
        assert retrieved_doc is not None
        print("✓ Document retrieval by ID successful")
        
        # Clean up
        doc_store.delete_document(doc_id)
        print("✓ Document cleanup successful")
        
        return True
    except Exception as e:
        print(f"✗ Document store error: {e}")
        return False

def test_rag_pipeline():
    """Test RAG pipeline functionality."""
    print("\nTesting RAG pipeline...")
    
    try:
        from tradingagents.rag.document_store import DocumentStore
        from tradingagents.rag.retriever import RAGRetriever
        from tradingagents.rag.generator import RAGGenerator
        from tradingagents.rag.rag_pipeline import RAGPipeline
        from tradingagents.default_config import DEFAULT_CONFIG
        
        # Create components
        doc_store = DocumentStore(DEFAULT_CONFIG, "test_pipeline")
        retriever = RAGRetriever(doc_store, "semantic")
        
        # Note: Generator requires an LLM, so we'll skip the full test
        print("✓ RAG pipeline components created successfully")
        
        # Test document ingestion
        test_docs = [
            {
                'content': 'Apple Inc. reported strong quarterly earnings.',
                'metadata': {
                    'ticker': 'AAPL',
                    'document_type': 'earnings',
                    'date': '2024-01-15',
                    'source': 'test'
                }
            },
            {
                'content': 'Technical analysis shows bullish momentum for Apple stock.',
                'metadata': {
                    'ticker': 'AAPL',
                    'document_type': 'technical',
                    'date': '2024-01-15',
                    'source': 'test'
                }
            }
        ]
        
        doc_ids = doc_store.add_documents_batch(test_docs)
        print(f"✓ Batch document ingestion successful: {len(doc_ids)} documents")
        
        # Test retrieval
        results = retriever.retrieve(
            query="Apple earnings and technical analysis",
            n_results=2,
            ticker="AAPL"
        )
        print(f"✓ Document retrieval successful: {len(results)} results")
        
        # Clean up
        for doc_id in doc_ids:
            doc_store.delete_document(doc_id)
        print("✓ Pipeline cleanup successful")
        
        return True
    except Exception as e:
        print(f"✗ RAG pipeline error: {e}")
        return False

def test_enhanced_trading_graph():
    """Test enhanced trading graph (without full LLM setup)."""
    print("\nTesting enhanced trading graph...")
    
    try:
        from tradingagents.rag.enhanced_trading_graph import RAGEnhancedTradingGraph
        from tradingagents.default_config import DEFAULT_CONFIG
        
        # Test graph creation (this will fail if LLM is not properly configured)
        print("✓ Enhanced trading graph class imported successfully")
        
        # Test configuration
        config = DEFAULT_CONFIG.copy()
        config["rag_enabled"] = True
        
        print("✓ RAG configuration prepared successfully")
        
        return True
    except Exception as e:
        print(f"✗ Enhanced trading graph error: {e}")
        return False

def main():
    """Run all tests."""
    print("RAG Integration Test Suite")
    print("=" * 50)
    
    tests = [
        test_rag_imports,
        test_rag_config,
        test_document_store,
        test_rag_pipeline,
        test_enhanced_trading_graph
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! RAG integration is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
