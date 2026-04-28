"""
Embedding generation and vector database management
"""

import os
from typing import List, Dict, Any, Optional
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings

from .utils import setup_logger

logger = setup_logger(__name__)


class EmbeddingManager:
    """Manage embeddings and vector database operations."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        persist_directory: str = "./data/chroma_db",
        collection_name: str = "pizzabot_knowledge"
    ):
        """
        Initialize embedding manager.
        
        Args:
            model_name: Name of sentence transformer model
            persist_directory: Directory for ChromaDB persistence
            collection_name: Name of ChromaDB collection
        """
        self.model_name = model_name
        self.persist_directory = persist_directory
        self.collection_name = collection_name
        
        # Load sentence transformer model
        logger.info(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        
        # Initialize ChromaDB
        os.makedirs(persist_directory, exist_ok=True)
        self.client = chromadb.Client(Settings(
            persist_directory=persist_directory,
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        try:
            self.collection = self.client.get_collection(collection_name)
            logger.info(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            logger.info(f"Created new collection: {collection_name}")
    
    def generate_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Generate embeddings for texts.
        
        Args:
            texts: List of text strings
            
        Returns:
            Array of embeddings
        """
        embeddings = self.model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32
        )
        return embeddings
    
    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Add documents to vector database.
        
        Args:
            documents: List of document texts
            metadatas: Optional metadata for each document
            ids: Optional IDs for each document
        """
        if not documents:
            logger.warning("No documents to add")
            return
        
        # Generate embeddings
        logger.info(f"Generating embeddings for {len(documents)} documents")
        embeddings = self.generate_embeddings(documents)
        
        # Generate IDs if not provided
        if ids is None:
            ids = [f"doc_{i}" for i in range(len(documents))]
        
        # Prepare metadatas
        if metadatas is None:
            metadatas = [{"source": "knowledge_base"} for _ in documents]
        
        # Add to collection
        self.collection.add(
            documents=documents,
            embeddings=embeddings.tolist(),
            metadatas=metadatas,
            ids=ids
        )
        
        logger.info(f"Added {len(documents)} documents to collection")
    
    def query(
        self,
        query_text: str,
        n_results: int = 3,
        where: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Query vector database for similar documents.
        
        Args:
            query_text: Query string
            n_results: Number of results to return
            where: Optional metadata filter
            
        Returns:
            Query results with documents and scores
        """
        # Generate query embedding
        query_embedding = self.generate_embeddings([query_text])[0]
        
        # Query collection
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results,
            where=where
        )
        
        return results
    
    def retrieve_context(
        self,
        query: str,
        top_k: int = 3,
        similarity_threshold: float = 0.7
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context for query.
        
        Args:
            query: User query
            top_k: Number of documents to retrieve
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of relevant documents with metadata
        """
        results = self.query(query, n_results=top_k)
        
        # Format results
        context_docs = []
        for i in range(len(results['documents'][0])):
            # Calculate similarity (1 - distance for cosine)
            similarity = 1 - results['distances'][0][i]
            
            if similarity >= similarity_threshold:
                context_docs.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'similarity': similarity,
                    'id': results['ids'][0][i]
                })
        
        logger.info(f"Retrieved {len(context_docs)} relevant documents for query")
        return context_docs
    
    def clear_collection(self) -> None:
        """Clear all documents from collection."""
        self.client.delete_collection(self.collection_name)
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info("Cleared collection")
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Get collection statistics.
        
        Returns:
            Dictionary with collection stats
        """
        count = self.collection.count()
        return {
            'name': self.collection_name,
            'count': count,
            'model': self.model_name
        }
