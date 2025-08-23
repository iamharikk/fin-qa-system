import numpy as np
import re
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from embedding_indexer import EmbeddingIndexer

class HybridRetriever:
    """
    Hybrid retrieval system combining dense (vector) and sparse (keyword) retrieval
    """
    
    def __init__(self, indexer: EmbeddingIndexer):
        """
        Initialize hybrid retriever with pre-built indexes
        
        Args:
            indexer: EmbeddingIndexer with loaded dense and sparse indexes
        """
        self.indexer = indexer
        self.embedding_model = indexer.embedding_model
        print("Hybrid retriever initialized")
    
    def preprocess_query(self, query: str) -> str:
        """
        Clean and preprocess user query
        
        Args:
            query: Raw user query string
            
        Returns:
            Cleaned query string
        """
        # Convert to lowercase
        query = query.lower()
        
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query.strip())
        
        # Remove special characters but keep alphanumeric and spaces
        query = re.sub(r'[^\w\s]', ' ', query)
        
        # Remove extra spaces again
        query = re.sub(r'\s+', ' ', query.strip())
        
        return query
    
    def dense_retrieval(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Retrieve chunks using dense vector similarity (FAISS)
        
        Args:
            query: User query string
            top_k: Number of top results to retrieve
            
        Returns:
            List of tuples (chunk_index, similarity_score)
        """
        # Create embedding for the query using same model as chunks
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Search in FAISS index for most similar vectors
        # search() returns distances and indices
        distances, indices = self.indexer.dense_index.search(
            query_embedding.astype('float32'), 
            top_k
        )
        
        # Convert L2 distances to similarity scores (lower distance = higher similarity)
        # Formula: similarity = 1 / (1 + distance)
        similarities = 1.0 / (1.0 + distances[0])
        
        # Return list of (index, similarity) tuples
        results = [(int(idx), float(sim)) for idx, sim in zip(indices[0], similarities)]
        
        return results
    
    def sparse_retrieval(self, query: str, top_k: int = 10) -> List[Tuple[int, float]]:
        """
        Retrieve chunks using sparse keyword matching (BM25)
        
        Args:
            query: User query string
            top_k: Number of top results to retrieve
            
        Returns:
            List of tuples (chunk_index, bm25_score)
        """
        # Tokenize query for BM25 (same way as documents were tokenized)
        query_tokens = query.lower().split()
        
        # Get BM25 scores for all documents
        bm25_scores = self.indexer.sparse_index.get_scores(query_tokens)
        
        # Get top-k indices sorted by score
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        
        # Return list of (index, score) tuples
        results = [(int(idx), float(bm25_scores[idx])) for idx in top_indices]
        
        return results
    
    def combine_results(self, dense_results: List[Tuple[int, float]], 
                       sparse_results: List[Tuple[int, float]], 
                       alpha: float = 0.5) -> List[Tuple[int, float]]:
        """
        Combine dense and sparse retrieval results using weighted fusion
        
        Args:
            dense_results: Results from dense retrieval
            sparse_results: Results from sparse retrieval  
            alpha: Weight for dense results (1-alpha for sparse results)
            
        Returns:
            Combined and sorted list of (chunk_index, combined_score)
        """
        # Create dictionaries for easy lookup
        dense_scores = {idx: score for idx, score in dense_results}
        sparse_scores = {idx: score for idx, score in sparse_results}
        
        # Get all unique chunk indices from both retrievals
        all_indices = set(dense_scores.keys()) | set(sparse_scores.keys())
        
        # Normalize scores to [0, 1] range for fair combination
        # Dense scores normalization
        if dense_scores:
            max_dense = max(dense_scores.values()) if dense_scores.values() else 1.0
            min_dense = min(dense_scores.values()) if dense_scores.values() else 0.0
            dense_range = max_dense - min_dense if max_dense != min_dense else 1.0
        
        # Sparse scores normalization  
        if sparse_scores:
            max_sparse = max(sparse_scores.values()) if sparse_scores.values() else 1.0
            min_sparse = min(sparse_scores.values()) if sparse_scores.values() else 0.0
            sparse_range = max_sparse - min_sparse if max_sparse != min_sparse else 1.0
        
        # Combine scores for each chunk
        combined_results = []
        for idx in all_indices:
            # Normalize dense score
            dense_score = dense_scores.get(idx, 0.0)
            if dense_scores:
                dense_norm = (dense_score - min_dense) / dense_range
            else:
                dense_norm = 0.0
                
            # Normalize sparse score
            sparse_score = sparse_scores.get(idx, 0.0)
            if sparse_scores:
                sparse_norm = (sparse_score - min_sparse) / sparse_range
            else:
                sparse_norm = 0.0
            
            # Weighted combination
            combined_score = alpha * dense_norm + (1 - alpha) * sparse_norm
            combined_results.append((idx, combined_score))
        
        # Sort by combined score (highest first)
        combined_results.sort(key=lambda x: x[1], reverse=True)
        
        return combined_results
    
    def retrieve(self, query: str, top_k: int = 5, alpha: float = 0.6) -> List[Dict[str, Any]]:
        """
        Main retrieval function using hybrid approach
        
        Args:
            query: User query string
            top_k: Number of final results to return
            alpha: Weight for dense retrieval (0.6 means 60% dense, 40% sparse)
            
        Returns:
            List of retrieved chunk dictionaries with scores
        """
        print(f"\nHybrid Retrieval for: '{query}'")
        
        # Step 1: Preprocess query
        processed_query = self.preprocess_query(query)
        print(f"Processed query: '{processed_query}'")
        
        # Step 2: Dense retrieval (vector similarity)
        print("Performing dense retrieval...")
        dense_results = self.dense_retrieval(processed_query, top_k=top_k*2)  # Get more for combination
        
        # Step 3: Sparse retrieval (keyword matching)
        print("Performing sparse retrieval...")
        sparse_results = self.sparse_retrieval(processed_query, top_k=top_k*2)  # Get more for combination
        
        # Step 4: Combine results using weighted fusion
        print("Combining results...")
        combined_results = self.combine_results(dense_results, sparse_results, alpha=alpha)
        
        # Step 5: Get top-k final results with chunk data
        final_results = []
        for i, (chunk_idx, score) in enumerate(combined_results[:top_k]):
            chunk_data = self.indexer.chunks_data[chunk_idx].copy()
            chunk_data['retrieval_score'] = score
            chunk_data['retrieval_rank'] = i + 1
            final_results.append(chunk_data)
        
        print(f"Retrieved {len(final_results)} chunks")
        return final_results
    
    def display_results(self, results: List[Dict[str, Any]], show_text_length: int = 200):
        """
        Display retrieval results in a readable format
        
        Args:
            results: List of retrieved chunk dictionaries
            show_text_length: Number of characters to show from chunk text
        """
        print(f"\n=== Retrieval Results (Top {len(results)}) ===")
        
        for i, result in enumerate(results, 1):
            print(f"\nRank {i} (Score: {result['retrieval_score']:.4f})")
            print(f"  Chunk ID: {result['chunk_id'][:8]}...")
            print(f"  Target Size: {result['chunk_size_target']} tokens")
            print(f"  Actual Size: {result['token_count']} tokens")
            print(f"  Text: {result['chunk_text'][:show_text_length]}...")
            print(f"  Original Q: {result['original_question'][:100]}...")

# Example usage and testing
if __name__ == "__main__":
    # Load pre-built indexes
    indexer = EmbeddingIndexer("all-MiniLM-L6-v2")
    
    try:
        # Load saved indexes
        indexer.load_indexes("../data/indexes")
        
        # Create hybrid retriever
        retriever = HybridRetriever(indexer)
        
        # Test queries
        test_queries = [
            "What was TCS revenue in 2025?",
            "TCS profit margin",
            "employee costs",
            "cash and bank balance",
            "total expenses"
        ]
        
        print("=== Testing Hybrid Retrieval ===")
        
        for query in test_queries:
            results = retriever.retrieve(query, top_k=3, alpha=0.6)
            retriever.display_results(results, show_text_length=150)
            print("="*60)
            
    except FileNotFoundError:
        print("Error: Index files not found.")
        print("Please run embedding_indexer.py first to create indexes.")