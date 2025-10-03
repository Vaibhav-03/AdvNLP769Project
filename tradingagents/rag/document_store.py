"""
Document Store for RAG Framework

Handles storage, indexing, and retrieval of financial documents and data.
Supports multiple document types and metadata filtering.
"""

import os
import json
import hashlib
from typing import List, Dict, Any, Optional, Union
from datetime import datetime
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import pandas as pd
from pathlib import Path


class DocumentStore:
    """Enhanced document store for financial documents with RAG capabilities."""
    
    def __init__(self, config: Dict[str, Any], collection_name: str = "financial_documents"):
        """
        Initialize the document store.
        
        Args:
            config: Configuration dictionary
            collection_name: Name of the ChromaDB collection
        """
        self.config = config
        self.collection_name = collection_name
        
        # Initialize embedding model
        if config["backend_url"] == "http://localhost:11434/v1":
            self.embedding_model = "nomic-embed-text"
        else:
            self.embedding_model = "text-embedding-3-small"
            
        self.client = OpenAI(base_url=config["backend_url"])
        
        # Initialize ChromaDB
        self.chroma_client = chromadb.Client(Settings(allow_reset=True))
        try:
            self.collection = self.chroma_client.get_collection(name=collection_name)
        except:
            self.collection = self.chroma_client.create_collection(name=collection_name)
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding for text using the configured model."""
        response = self.client.embeddings.create(
            model=self.embedding_model,
            input=text
        )
        return response.data[0].embedding
    
    def add_document(self, 
                    content: str, 
                    metadata: Dict[str, Any],
                    document_id: Optional[str] = None) -> str:
        """
        Add a document to the store.
        
        Args:
            content: Document content
            metadata: Document metadata (ticker, date, type, etc.)
            document_id: Optional custom document ID
            
        Returns:
            Document ID
        """
        if document_id is None:
            # Generate ID from content hash
            content_hash = hashlib.md5(content.encode()).hexdigest()[:16]
            document_id = f"{metadata.get('ticker', 'unknown')}_{content_hash}"
        
        # Generate embedding
        embedding = self.get_embedding(content)
        
        # Add to collection
        self.collection.add(
            documents=[content],
            metadatas=[metadata],
            embeddings=[embedding],
            ids=[document_id]
        )
        
        return document_id
    
    def add_documents_batch(self, 
                           documents: List[Dict[str, Any]]) -> List[str]:
        """
        Add multiple documents in batch.
        
        Args:
            documents: List of dicts with 'content', 'metadata', and optional 'id'
            
        Returns:
            List of document IDs
        """
        contents = [doc['content'] for doc in documents]
        metadatas = [doc['metadata'] for doc in documents]
        ids = [doc.get('id', f"doc_{i}") for i, doc in enumerate(documents)]
        
        # Generate embeddings
        embeddings = [self.get_embedding(content) for content in contents]
        
        # Add to collection
        self.collection.add(
            documents=contents,
            metadatas=metadatas,
            embeddings=embeddings,
            ids=ids
        )
        
        return ids
    
    def search_documents(self, 
                        query: str, 
                        n_results: int = 5,
                        filter_metadata: Optional[Dict[str, Any]] = None,
                        ticker: Optional[str] = None,
                        document_type: Optional[str] = None,
                        date_range: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant documents.
        
        Args:
            query: Search query
            n_results: Number of results to return
            filter_metadata: Additional metadata filters
            ticker: Filter by ticker symbol
            document_type: Filter by document type
            date_range: Filter by date range (start_date, end_date)
            
        Returns:
            List of matching documents with metadata and scores
        """
        # Build where clause for filtering
        where_clause = {}
        if filter_metadata:
            where_clause.update(filter_metadata)
        if ticker:
            where_clause['ticker'] = ticker
        if document_type:
            where_clause['document_type'] = document_type
        if date_range:
            where_clause['date'] = {"$gte": date_range[0], "$lte": date_range[1]}
        
        # Generate query embedding
        query_embedding = self.get_embedding(query)
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where=where_clause if where_clause else None,
            include=["metadatas", "documents", "distances"]
        )
        
        # Format results
        matched_documents = []
        for i in range(len(results["documents"][0])):
            matched_documents.append({
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "similarity_score": 1 - results["distances"][0][i],
                "document_id": results["ids"][0][i]
            })
        
        return matched_documents
    
    def get_document_by_id(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID."""
        try:
            result = self.collection.get(ids=[document_id], include=["metadatas", "documents"])
            if result["documents"]:
                return {
                    "content": result["documents"][0],
                    "metadata": result["metadatas"][0],
                    "document_id": document_id
                }
        except Exception as e:
            print(f"Error retrieving document {document_id}: {e}")
        return None
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document by ID."""
        try:
            self.collection.delete(ids=[document_id])
            return True
        except Exception as e:
            print(f"Error deleting document {document_id}: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection."""
        count = self.collection.count()
        return {
            "total_documents": count,
            "collection_name": self.collection_name
        }
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
            self.collection = self.chroma_client.create_collection(name=self.collection_name)
            return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False
