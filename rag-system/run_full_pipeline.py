#!/usr/bin/env python3
"""
Complete RAG pipeline runner
Executes all steps: data processing -> indexing -> RAG system
"""

import os
import sys
from data_processor import DataProcessor
from embedding_indexer import EmbeddingIndexer
from response_generator import ResponseGenerator

def run_complete_pipeline():
    """
    Run the complete RAG pipeline from start to finish
    """
    print("="*70)
    print("COMPLETE RAG PIPELINE EXECUTION")
    print("="*70)
    
    try:
        # Step 1: Data Processing
        print("\n--- STEP 1: DATA PROCESSING ---")
        processor = DataProcessor("../data/tcs_qa_dataset.csv")
        chunks = processor.process_qa_data(chunk_sizes=[100, 400])
        processor.save_chunks(chunks, "../data/processed_chunks.csv")
        print("Data processing completed successfully!")
        
        # Step 2: Embedding and Indexing
        print("\n--- STEP 2: EMBEDDING & INDEXING ---")
        indexer = EmbeddingIndexer("all-MiniLM-L6-v2")
        indexer.create_indexes("../data/processed_chunks.csv")
        indexer.save_indexes("../data/indexes")
        print("Embedding and indexing completed successfully!")
        
        # Step 3: Test Complete RAG System
        print("\n--- STEP 3: RAG SYSTEM TESTING ---")
        rag_system = ResponseGenerator(indexer)
        
        # Test questions
        test_questions = [
            "What was TCS revenue in 2025?",
            "How much profit did TCS make in 2024?",
            "What are TCS main expenses?"
        ]
        
        for query in test_questions:
            print(f"\n{'-'*50}")
            result = rag_system.complete_rag_pipeline(query)
            
            # Display concise result
            print(f"\nQUESTION: {result['query']}")
            print(f"ANSWER: {result['generated_answer']}")
        
        print(f"\n{'='*70}")
        print("COMPLETE RAG PIPELINE FINISHED SUCCESSFULLY!")
        print("="*70)
        print("\nTo run interactive session:")
        print("python response_generator.py")
        print("Then uncomment the interactive_rag_session() call")
        
    except Exception as e:
        print(f"\nError in pipeline: {e}")
        print("Please check that:")
        print("1. TCS dataset exists at ../data/tcs_qa_dataset.csv")
        print("2. All requirements are installed: pip install -r ../requirements.txt")
        sys.exit(1)

if __name__ == "__main__":
    run_complete_pipeline()