import torch
from typing import List, Dict, Any, Tuple
from sentence_transformers import CrossEncoder
from hybrid_retriever import HybridRetriever
from embedding_indexer import EmbeddingIndexer

class AdvancedRAG:
    """
    Advanced RAG system with multi-stage retrieval and re-ranking
    
    Stage 1: Broad retrieval using hybrid approach (dense + sparse)
    Stage 2: Re-ranking using cross-encoder for precise relevance scoring
    """
    
    def __init__(self, indexer: EmbeddingIndexer, cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize advanced RAG with hybrid retriever and cross-encoder
        
        Args:
            indexer: EmbeddingIndexer with loaded indexes
            cross_encoder_model: Pre-trained cross-encoder model for re-ranking
        """
        # Initialize hybrid retriever for stage 1
        self.hybrid_retriever = HybridRetriever(indexer)
        
        # Load cross-encoder model for stage 2 re-ranking
        print(f"Loading cross-encoder model: {cross_encoder_model}")
        self.cross_encoder = CrossEncoder(cross_encoder_model)
        
        print("Advanced RAG system initialized")
    
    def stage1_broad_retrieval(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Stage 1: Broad retrieval using hybrid approach
        Retrieves more candidates than needed for re-ranking
        
        Args:
            query: User query string
            top_k: Number of candidates to retrieve (should be larger than final results)
            
        Returns:
            List of candidate chunks from hybrid retrieval
        """
        print(f"\n=== Stage 1: Broad Retrieval ===")
        print(f"Retrieving {top_k} candidates for query: '{query}'")
        
        # Use hybrid retrieval to get broad set of candidates
        candidates = self.hybrid_retriever.retrieve(query, top_k=top_k, alpha=0.6)
        
        print(f"Retrieved {len(candidates)} candidates from Stage 1")
        return candidates
    
    def stage2_reranking(self, query: str, candidates: List[Dict[str, Any]], final_k: int = 5) -> List[Dict[str, Any]]:
        """
        Stage 2: Re-rank candidates using cross-encoder for precise scoring
        
        Args:
            query: User query string
            candidates: List of candidate chunks from Stage 1
            final_k: Number of final results to return after re-ranking
            
        Returns:
            Re-ranked list of top chunks
        """
        print(f"\n=== Stage 2: Cross-Encoder Re-ranking ===")
        print(f"Re-ranking {len(candidates)} candidates to get top {final_k}")
        
        if not candidates:
            return []
        
        # Prepare query-document pairs for cross-encoder
        # Cross-encoder takes (query, document) pairs and outputs relevance scores
        query_doc_pairs = []
        for candidate in candidates:
            # Use chunk text as document
            doc_text = candidate['chunk_text']
            query_doc_pairs.append([query, doc_text])
        
        # Get relevance scores from cross-encoder
        # Cross-encoder gives more accurate relevance than bi-encoder similarity
        print("Computing cross-encoder relevance scores...")
        relevance_scores = self.cross_encoder.predict(query_doc_pairs)
        
        # Add cross-encoder scores to candidates
        for i, candidate in enumerate(candidates):
            candidate['cross_encoder_score'] = float(relevance_scores[i])
            candidate['stage1_score'] = candidate.get('retrieval_score', 0.0)  # Keep original score
        
        # Sort by cross-encoder score (higher is better)
        reranked_candidates = sorted(candidates, key=lambda x: x['cross_encoder_score'], reverse=True)
        
        # Update ranks based on re-ranking
        for i, candidate in enumerate(reranked_candidates[:final_k]):
            candidate['final_rank'] = i + 1
        
        final_results = reranked_candidates[:final_k]
        print(f"Re-ranking completed. Returning top {len(final_results)} results")
        
        return final_results
    
    def multi_stage_retrieve(self, query: str, stage1_k: int = 20, final_k: int = 5) -> List[Dict[str, Any]]:
        """
        Complete multi-stage retrieval pipeline
        
        Args:
            query: User query string
            stage1_k: Number of candidates to retrieve in Stage 1
            final_k: Number of final results after re-ranking
            
        Returns:
            Final re-ranked results
        """
        print(f"\n" + "="*60)
        print(f"MULTI-STAGE RETRIEVAL: '{query}'")
        print(f"="*60)
        
        # Stage 1: Broad retrieval using hybrid approach
        candidates = self.stage1_broad_retrieval(query, top_k=stage1_k)
        
        # Stage 2: Re-ranking using cross-encoder
        final_results = self.stage2_reranking(query, candidates, final_k=final_k)
        
        return final_results
    
    def display_advanced_results(self, results: List[Dict[str, Any]], show_text_length: int = 200):
        """
        Display advanced retrieval results with both stage scores
        
        Args:
            results: List of final ranked results
            show_text_length: Number of characters to show from chunk text
        """
        print(f"\n=== FINAL RESULTS (Top {len(results)}) ===")
        
        for result in results:
            print(f"\nRank {result['final_rank']}")
            print(f"  Cross-Encoder Score: {result['cross_encoder_score']:.4f}")
            print(f"  Stage 1 Score: {result['stage1_score']:.4f}")
            print(f"  Chunk ID: {result['chunk_id'][:8]}...")
            print(f"  Target Size: {result['chunk_size_target']} tokens")
            print(f"  Text: {result['chunk_text'][:show_text_length]}...")
            print(f"  Original Q: {result['original_question'][:100]}...")
    
    def compare_retrieval_stages(self, query: str, stage1_k: int = 10, final_k: int = 5):
        """
        Compare results before and after re-ranking to show improvement
        
        Args:
            query: User query string
            stage1_k: Number of Stage 1 candidates
            final_k: Number of final results
        """
        print(f"\n" + "="*70)
        print(f"RETRIEVAL COMPARISON: '{query}'")
        print(f"="*70)
        
        # Get Stage 1 results
        stage1_results = self.stage1_broad_retrieval(query, top_k=stage1_k)
        
        # Get Stage 2 (final) results
        stage2_results = self.stage2_reranking(query, stage1_results, final_k=final_k)
        
        print(f"\n--- STAGE 1 RESULTS (Top {final_k}) ---")
        for i, result in enumerate(stage1_results[:final_k], 1):
            print(f"{i}. Score: {result['retrieval_score']:.4f} | Text: {result['chunk_text'][:100]}...")
        
        print(f"\n--- STAGE 2 RESULTS (After Re-ranking) ---")
        for result in stage2_results:
            print(f"{result['final_rank']}. Cross-Encoder: {result['cross_encoder_score']:.4f} | Text: {result['chunk_text'][:100]}...")
        
        return stage2_results

# Example usage and testing
if __name__ == "__main__":
    # Load pre-built indexes
    indexer = EmbeddingIndexer("all-MiniLM-L6-v2")
    
    try:
        # Load saved indexes
        indexer.load_indexes("../data/indexes")
        
        # Create advanced RAG system
        advanced_rag = AdvancedRAG(indexer)
        
        # Test queries
        test_queries = [
            "What was TCS revenue in 2025?",
            "How much profit did TCS make?",
            "TCS employee expenses",
        ]
        
        print("=== Testing Advanced Multi-Stage RAG ===")
        
        for query in test_queries:
            # Compare before and after re-ranking
            results = advanced_rag.compare_retrieval_stages(query, stage1_k=10, final_k=3)
            print("\n" + "="*70)
            
    except FileNotFoundError:
        print("Error: Index files not found.")
        print("Please run embedding_indexer.py first to create indexes.")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure all required packages are installed.")