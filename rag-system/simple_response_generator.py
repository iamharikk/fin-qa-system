import re
import time
from typing import Dict, Any, List
from advanced_rag import AdvancedRAG
from embedding_indexer import EmbeddingIndexer
from guardrails import RAGGuardrails

class SimpleResponseGenerator:
    """
    Simple response generator that extracts answers directly from retrieved context
    More reliable than small language models for factual financial data
    """
    
    def __init__(self, indexer: EmbeddingIndexer):
        """
        Initialize simple response generator with advanced RAG
        
        Args:
            indexer: EmbeddingIndexer with loaded indexes
        """
        # Initialize advanced RAG system for retrieval
        self.advanced_rag = AdvancedRAG(indexer)
        self.guardrails = RAGGuardrails()
        
        print("Simple response generator initialized")
    
    def extract_answer_from_context(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Extract answer directly from retrieved context using pattern matching
        
        Args:
            query: User's question
            retrieved_chunks: Retrieved context chunks
            
        Returns:
            Extracted answer string
        """
        if not retrieved_chunks:
            return "No relevant information found."
        
        # Combine all retrieved text
        combined_context = " ".join([chunk['chunk_text'] for chunk in retrieved_chunks])
        
        # Look for the most relevant chunk (highest score)
        best_chunk = retrieved_chunks[0]['chunk_text']
        
        # Simple pattern matching for financial queries
        query_lower = query.lower()
        
        # Check if the answer is directly in the best matching chunk
        if 'rs' in best_chunk.lower() and any(word in query_lower for word in ['revenue', 'income', 'profit', 'expense', 'cost']):
            # Extract financial figures and context
            sentences = re.split(r'[.!?]', best_chunk)
            for sentence in sentences:
                if any(word in sentence.lower() for word in query_lower.split()):
                    if 'rs' in sentence.lower() or 'crore' in sentence.lower():
                        return sentence.strip()
        
        # Fallback: return the most relevant sentence from best chunk
        sentences = re.split(r'[.!?]', best_chunk)
        for sentence in sentences:
            # Look for sentences containing key query terms
            query_words = set(query_lower.split())
            sentence_words = set(sentence.lower().split())
            
            # If sentence contains multiple query words, it's likely the answer
            if len(query_words.intersection(sentence_words)) >= 2:
                return sentence.strip()
        
        # Final fallback: return first part of best chunk
        return best_chunk[:200] + "..." if len(best_chunk) > 200 else best_chunk
    
    def complete_rag_pipeline(self, query: str, stage1_k: int = 15, final_k: int = 3) -> Dict[str, Any]:
        """
        Complete end-to-end RAG pipeline: retrieve + extract answer
        
        Args:
            query: User's question
            stage1_k: Number of candidates for Stage 1 retrieval
            final_k: Number of final chunks for answer extraction
            
        Returns:
            Dictionary with query, retrieved chunks, and extracted answer
        """
        start_time = time.time()
        
        print(f"\n{'='*70}")
        print(f"SIMPLE RAG PIPELINE: '{query}'")
        print(f"{'='*70}")
        
        # Step 1: Multi-stage retrieval
        print("\n--- STEP 1: RETRIEVAL ---")
        retrieved_chunks = self.advanced_rag.multi_stage_retrieve(
            query, stage1_k=stage1_k, final_k=final_k
        )
        
        # Step 2: Extract answer directly from context
        print(f"\n--- STEP 2: ANSWER EXTRACTION ---")
        extracted_answer = self.extract_answer_from_context(query, retrieved_chunks)
        
        processing_time = time.time() - start_time
        
        # Apply guardrails
        validation_result = self.guardrails.apply_guardrails(query, extracted_answer, retrieved_chunks)
        
        # Determine final answer
        if validation_result['overall_valid']:
            final_answer = extracted_answer
            confidence = validation_result['final_confidence']
            method = "Context Extraction RAG"
        else:
            final_answer = self.guardrails.create_safe_response(validation_result)
            confidence = 0.2
            method = "Guardrail Filtered"
        
        # Prepare final result
        result = {
            'query': query,
            'retrieved_chunks': retrieved_chunks,
            'extracted_answer': final_answer,
            'confidence_score': confidence,
            'method_used': method,
            'response_time': processing_time,
            'validation_issues': validation_result.get('all_issues', [])
        }
        
        return result
    
    def display_complete_result(self, result: Dict[str, Any]):
        """
        Display the complete RAG result in a readable format
        
        Args:
            result: Result dictionary from complete_rag_pipeline
        """
        print(f"\n{'='*70}")
        print(f"FINAL RAG RESULT")
        print(f"{'='*70}")
        
        print(f"\nQUESTION: {result['query']}")
        
        print(f"\nRETRIEVED CONTEXT:")
        for i, chunk in enumerate(result['retrieved_chunks'], 1):
            print(f"{i}. [Score: {chunk['cross_encoder_score']:.4f}] {chunk['chunk_text'][:200]}...")
        
        print(f"\nEXTRACTED ANSWER:")
        print(f"{result['extracted_answer']}")
        
        print(f"\nCONFIDENCE: {result['confidence_score']:.2f}")
        print(f"METHOD: {result['method_used']}")
        print(f"TIME: {result['response_time']:.2f}s")
    
    def interactive_rag_session(self):
        """
        Start an interactive RAG Q&A session
        """
        print(f"\n{'='*70}")
        print("INTERACTIVE SIMPLE RAG FINANCIAL Q&A SYSTEM")
        print("Ask questions about TCS financial data")
        print("Type 'quit', 'exit', or 'q' to stop")
        print(f"{'='*70}")
        
        while True:
            user_query = input("\nYour question: ").strip()
            
            if user_query.lower() in ['quit', 'exit', 'q']:
                print("Thank you for using the RAG system!")
                break
            
            if not user_query:
                print("Please enter a question.")
                continue
            
            try:
                # Run complete RAG pipeline
                result = self.complete_rag_pipeline(user_query)
                
                # Display result
                self.display_complete_result(result)
                
            except Exception as e:
                print(f"Error processing query: {e}")
                print("Please try again with a different question.")

# Example usage and testing
if __name__ == "__main__":
    # Load pre-built indexes
    indexer = EmbeddingIndexer("all-MiniLM-L6-v2")
    
    try:
        # Get correct path for indexes
        import os
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        indexes_path = os.path.join(project_root, "data", "indexes")
        
        # Load saved indexes
        indexer.load_indexes(indexes_path)
        
        # Create simple RAG system
        rag_system = SimpleResponseGenerator(indexer)
        
        # Test with sample questions
        test_questions = [
            "What was TCS other income in 2025?",
            "TCS revenue in 2025",
            "How much profit did TCS make in 2024?",
            "TCS employee costs"
        ]
        
        print("=== Testing Simple RAG System ===")
        
        for query in test_questions:
            result = rag_system.complete_rag_pipeline(query)
            rag_system.display_complete_result(result)
            print("\n" + "="*70)
        
        # Uncomment to start interactive session
        # rag_system.interactive_rag_session()
        
    except FileNotFoundError:
        print("Error: Index files not found.")
        print("Please run the full pipeline first:")
        print("python run_full_pipeline.py")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure all required packages are installed.")