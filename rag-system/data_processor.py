import pandas as pd
import re
import uuid
from typing import List, Dict, Any

class DataProcessor:
    """
    Simple data processor to split Q&A pairs into chunks for RAG system
    """
    
    def __init__(self, csv_file_path: str):
        """
        Initialize with the TCS Q&A dataset
        
        Args:
            csv_file_path: Path to CSV file with Question and Answer columns
        """
        self.df = pd.read_csv(csv_file_path)
        print(f"Loaded {len(self.df)} Q&A pairs")
        
    def clean_text(self, text: str) -> str:
        """
        Clean text by removing extra spaces and normalizing
        
        Args:
            text: Input text to clean
            
        Returns:
            Cleaned text string
        """
        # Remove extra whitespaces and normalize
        text = re.sub(r'\s+', ' ', text.strip())
        return text
    
    def count_tokens(self, text: str) -> int:
        """
        Simple token counting by splitting on spaces
        (In real systems, use proper tokenizers like tiktoken)
        
        Args:
            text: Text to count tokens for
            
        Returns:
            Number of tokens (words)
        """
        return len(text.split())
    
    def create_chunks(self, text: str, chunk_size: int, overlap: int = 20) -> List[str]:
        """
        Split text into chunks of specified size with overlap
        
        Args:
            text: Text to split into chunks
            chunk_size: Target size for each chunk (in tokens)
            overlap: Number of overlapping tokens between chunks
            
        Returns:
            List of text chunks
        """
        words = text.split()
        chunks = []
        
        # If text is shorter than chunk size, return as single chunk
        if len(words) <= chunk_size:
            return [text]
        
        # Create overlapping chunks
        start = 0
        while start < len(words):
            # Get chunk of words
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            chunks.append(chunk_text)
            
            # Move start position with overlap
            start = end - overlap
            if start >= len(words):
                break
                
        return chunks
    
    def process_qa_data(self, chunk_sizes: List[int] = [100, 400]) -> List[Dict[str, Any]]:
        """
        Process Q&A data into chunks with metadata
        
        Args:
            chunk_sizes: List of different chunk sizes to create
            
        Returns:
            List of chunk dictionaries with metadata
        """
        all_chunks = []
        
        print(f"Processing data with chunk sizes: {chunk_sizes}")
        
        # Process each Q&A pair
        for idx, row in self.df.iterrows():
            question = self.clean_text(row['Question'])
            answer = self.clean_text(row['Answer'])
            
            # Combine question and answer as context
            full_text = f"Question: {question} Answer: {answer}"
            
            # Create chunks for each specified size
            for chunk_size in chunk_sizes:
                chunks = self.create_chunks(full_text, chunk_size)
                
                # Add each chunk with metadata
                for chunk_idx, chunk in enumerate(chunks):
                    chunk_data = {
                        'chunk_id': str(uuid.uuid4()),  # Unique ID for each chunk
                        'original_qa_index': idx,       # Reference to original Q&A pair
                        'chunk_size_target': chunk_size, # Target chunk size used
                        'chunk_index': chunk_idx,       # Index within this Q&A pair
                        'chunk_text': chunk,            # The actual text chunk
                        'token_count': self.count_tokens(chunk), # Actual token count
                        'original_question': question,  # Original question for reference
                        'original_answer': answer,      # Original answer for reference
                        'source': 'TCS_Financial_Data'  # Data source identifier
                    }
                    all_chunks.append(chunk_data)
        
        print(f"Created {len(all_chunks)} total chunks")
        
        # Print summary statistics
        for size in chunk_sizes:
            count = len([c for c in all_chunks if c['chunk_size_target'] == size])
            print(f"  - {count} chunks with target size {size} tokens")
        
        return all_chunks
    
    def save_chunks(self, chunks: List[Dict[str, Any]], output_file: str):
        """
        Save processed chunks to CSV file
        
        Args:
            chunks: List of chunk dictionaries
            output_file: Output CSV file path
        """
        chunks_df = pd.DataFrame(chunks)
        chunks_df.to_csv(output_file, index=False)
        print(f"Saved {len(chunks)} chunks to {output_file}")
    
    def display_sample_chunks(self, chunks: List[Dict[str, Any]], num_samples: int = 3):
        """
        Display sample chunks for inspection
        
        Args:
            chunks: List of chunk dictionaries
            num_samples: Number of samples to display
        """
        print(f"\n=== Sample Chunks (showing {num_samples}) ===")
        
        for i, chunk in enumerate(chunks[:num_samples]):
            print(f"\nChunk {i+1}:")
            print(f"  ID: {chunk['chunk_id'][:8]}...")  # Show first 8 chars
            print(f"  Target Size: {chunk['chunk_size_target']} tokens")
            print(f"  Actual Size: {chunk['token_count']} tokens")
            print(f"  Text: {chunk['chunk_text'][:200]}...")  # Show first 200 chars
            print(f"  Source Q: {chunk['original_question'][:100]}...")

# Example usage and testing
if __name__ == "__main__":
    # Initialize data processor
    processor = DataProcessor("../data/tcs_qa_dataset.csv")
    
    # Process data into chunks with two different sizes
    chunks = processor.process_qa_data(chunk_sizes=[100, 400])
    
    # Display sample chunks
    processor.display_sample_chunks(chunks)
    
    # Save processed chunks
    processor.save_chunks(chunks, "../data/processed_chunks.csv")
    
    print(f"\nData processing completed!")
    print(f"   Total chunks created: {len(chunks)}")
    print(f"   Saved to: ../data/processed_chunks.csv")