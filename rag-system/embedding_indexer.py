import pandas as pd
import numpy as np
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
from typing import List, Dict, Any, Tuple

class EmbeddingIndexer:
    """
    Creates dense (FAISS) and sparse (BM25) indexes for chunk retrieval
    """
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize the embedding model and indexer
        
        Args:
            model_name: Name of sentence transformer model to use
        """
        print(f"Loading embedding model: {model_name}")
        # Load the sentence transformer model for creating embeddings
        self.embedding_model = SentenceTransformer(model_name)
        
        # Get embedding dimension from the model
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        print(f"Embedding dimension: {self.embedding_dim}")
        
        # Initialize empty indexes (will be created during indexing)
        self.dense_index = None  # FAISS vector store
        self.sparse_index = None  # BM25 keyword index
        self.chunks_data = []    # Store original chunk data
        self.chunk_embeddings = None  # Store embeddings array
        
    def load_chunks(self, chunks_file: str) -> List[Dict[str, Any]]:
        """
        Load processed chunks from CSV file
        
        Args:
            chunks_file: Path to processed chunks CSV
            
        Returns:
            List of chunk dictionaries
        """
        print(f"Loading chunks from: {chunks_file}")
        chunks_df = pd.read_csv(chunks_file)
        
        # Convert DataFrame to list of dictionaries
        chunks = chunks_df.to_dict('records')
        print(f"Loaded {len(chunks)} chunks")
        
        return chunks
    
    def create_embeddings(self, chunks: List[Dict[str, Any]]) -> np.ndarray:
        """
        Create dense embeddings for all chunks using sentence transformer
        
        Args:
            chunks: List of chunk dictionaries with 'chunk_text' field
            
        Returns:
            NumPy array of embeddings (num_chunks x embedding_dim)
        """
        print("Creating dense embeddings...")
        
        # Extract text from chunks
        chunk_texts = [chunk['chunk_text'] for chunk in chunks]
        
        # Generate embeddings in batches for efficiency
        # encode() method converts text to vector representations
        embeddings = self.embedding_model.encode(
            chunk_texts,
            batch_size=32,  # Process 32 texts at once
            show_progress_bar=True,  # Show progress during encoding
            convert_to_numpy=True   # Return as NumPy array
        )
        
        print(f"Created embeddings shape: {embeddings.shape}")
        return embeddings
    
    def build_dense_index(self, embeddings: np.ndarray):
        """
        Build FAISS dense vector index for similarity search
        
        Args:
            embeddings: NumPy array of embeddings to index
        """
        print("Building FAISS dense index...")
        
        # Create FAISS index for L2 (Euclidean) distance
        # IndexFlatL2 does exhaustive search (good for small datasets)
        self.dense_index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Add all embeddings to the index
        self.dense_index.add(embeddings.astype('float32'))
        
        print(f"Dense index built with {self.dense_index.ntotal} vectors")
    
    def build_sparse_index(self, chunks: List[Dict[str, Any]]):
        """
        Build BM25 sparse index for keyword-based retrieval
        
        Args:
            chunks: List of chunk dictionaries with 'chunk_text' field
        """
        print("Building BM25 sparse index...")
        
        # Extract and tokenize text for BM25
        # BM25 works with tokenized text (list of words)
        chunk_texts = []
        for chunk in chunks:
            # Simple tokenization by splitting on spaces and converting to lowercase
            tokens = chunk['chunk_text'].lower().split()
            chunk_texts.append(tokens)
        
        # Create BM25 index with tokenized texts
        self.sparse_index = BM25Okapi(chunk_texts)
        
        print(f"BM25 index built with {len(chunk_texts)} documents")
    
    def create_indexes(self, chunks_file: str):
        """
        Complete indexing pipeline: load chunks, create embeddings, build indexes
        
        Args:
            chunks_file: Path to processed chunks CSV file
        """
        # Step 1: Load chunks data
        self.chunks_data = self.load_chunks(chunks_file)
        
        # Step 2: Create dense embeddings
        self.chunk_embeddings = self.create_embeddings(self.chunks_data)
        
        # Step 3: Build dense vector index (FAISS)
        self.build_dense_index(self.chunk_embeddings)
        
        # Step 4: Build sparse keyword index (BM25)
        self.build_sparse_index(self.chunks_data)
        
        print("All indexes created successfully!")
    
    def save_indexes(self, save_dir: str):
        """
        Save all indexes and data to disk for later loading
        
        Args:
            save_dir: Directory to save index files
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save FAISS dense index
        faiss.write_index(self.dense_index, os.path.join(save_dir, "dense_index.faiss"))
        
        # Save BM25 sparse index using pickle
        with open(os.path.join(save_dir, "sparse_index.pkl"), "wb") as f:
            pickle.dump(self.sparse_index, f)
        
        # Save chunk embeddings
        np.save(os.path.join(save_dir, "chunk_embeddings.npy"), self.chunk_embeddings)
        
        # Save chunks data
        with open(os.path.join(save_dir, "chunks_data.pkl"), "wb") as f:
            pickle.dump(self.chunks_data, f)
        
        print(f"Indexes saved to: {save_dir}")
    
    def load_indexes(self, save_dir: str):
        """
        Load previously saved indexes from disk
        
        Args:
            save_dir: Directory containing saved index files
        """
        # Load FAISS dense index
        self.dense_index = faiss.read_index(os.path.join(save_dir, "dense_index.faiss"))
        
        # Load BM25 sparse index
        with open(os.path.join(save_dir, "sparse_index.pkl"), "rb") as f:
            self.sparse_index = pickle.load(f)
        
        # Load chunk embeddings
        self.chunk_embeddings = np.load(os.path.join(save_dir, "chunk_embeddings.npy"))
        
        # Load chunks data
        with open(os.path.join(save_dir, "chunks_data.pkl"), "rb") as f:
            self.chunks_data = pickle.load(f)
        
        print(f"Indexes loaded from: {save_dir}")
    
    def get_index_stats(self):
        """
        Print statistics about created indexes
        """
        print("\n=== Index Statistics ===")
        if self.dense_index:
            print(f"Dense Index (FAISS): {self.dense_index.ntotal} vectors")
        if self.sparse_index:
            print(f"Sparse Index (BM25): {len(self.sparse_index.corpus_size)} documents")
        if self.chunks_data:
            print(f"Total Chunks: {len(self.chunks_data)}")
            # Show chunk size distribution
            size_100 = len([c for c in self.chunks_data if c['chunk_size_target'] == 100])
            size_400 = len([c for c in self.chunks_data if c['chunk_size_target'] == 400])
            print(f"  - 100-token chunks: {size_100}")
            print(f"  - 400-token chunks: {size_400}")

# Example usage and testing
if __name__ == "__main__":
    # Initialize indexer with MiniLM model
    indexer = EmbeddingIndexer("all-MiniLM-L6-v2")
    
    # Create indexes from processed chunks
    # First run data_processor.py to create the chunks file
    chunks_file = "../data/processed_chunks.csv"
    
    try:
        indexer.create_indexes(chunks_file)
        
        # Show index statistics
        indexer.get_index_stats()
        
        # Save indexes for later use
        indexer.save_indexes("../data/indexes")
        
        print("\nEmbedding and indexing completed!")
        
    except FileNotFoundError:
        print(f"Error: {chunks_file} not found.")
        print("Please run data_processor.py first to create processed chunks.")