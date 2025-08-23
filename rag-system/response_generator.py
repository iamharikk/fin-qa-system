import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List, Dict, Any
from advanced_rag import AdvancedRAG
from embedding_indexer import EmbeddingIndexer

class ResponseGenerator:
    """
    Complete RAG system with response generation using DistilGPT2
    Combines retrieval with text generation for final answers
    """
    
    def __init__(self, indexer: EmbeddingIndexer, model_name: str = "distilgpt2"):
        """
        Initialize response generator with advanced RAG and language model
        
        Args:
            indexer: EmbeddingIndexer with loaded indexes
            model_name: Name of the generative model to use
        """
        # Initialize advanced RAG system for retrieval
        self.advanced_rag = AdvancedRAG(indexer)
        
        # Load tokenizer and language model for generation
        print(f"Loading generative model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generator_model = AutoModelForCausalLM.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Set model to evaluation mode
        self.generator_model.eval()
        
        # Model context window limit (DistilGPT2 has 1024 tokens)
        self.max_context_length = 1024
        self.max_generation_length = 150  # Leave room for retrieved context
        
        print("Response generator initialized")
    
    def create_prompt_template(self, query: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """
        Create a formatted prompt combining query and retrieved context
        
        Args:
            query: User's question
            retrieved_chunks: List of relevant chunks from retrieval
            
        Returns:
            Formatted prompt string for the language model
        """
        # Create a simple, effective prompt for DistilGPT2
        prompt = "Based on TCS financial data:\n\n"
        
        # Add retrieved context (keep it concise)
        for i, chunk in enumerate(retrieved_chunks[:2], 1):  # Limit to top 2 chunks
            chunk_text = chunk['chunk_text'][:300]  # Limit chunk length
            prompt += f"Fact {i}: {chunk_text}\n"
        
        # Add the question and answer prompt
        prompt += f"\nQuestion: {query}\nAnswer: TCS"
        
        return prompt
    
    def limit_context_tokens(self, prompt: str) -> str:
        """
        Limit prompt to fit within model's context window
        
        Args:
            prompt: Full prompt string
            
        Returns:
            Truncated prompt that fits within token limit
        """
        # Tokenize the prompt to check length
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        
        # Calculate available tokens (total - generation space)
        available_tokens = self.max_context_length - self.max_generation_length
        
        # Truncate if necessary
        if len(tokens) > available_tokens:
            # Keep the beginning (instruction) and end (question)
            truncated_tokens = tokens[:available_tokens-10]  # Leave some buffer
            prompt = self.tokenizer.decode(truncated_tokens, skip_special_tokens=True)
            prompt += "\nAnswer:"  # Ensure it ends with answer prompt
            
            print(f"Prompt truncated to fit {available_tokens} tokens")
        
        return prompt
    
    def generate_response(self, prompt: str) -> str:
        """
        Generate response using the language model
        
        Args:
            prompt: Formatted prompt with context and question
            
        Returns:
            Generated answer string
        """
        # Tokenize input prompt with attention mask
        inputs = self.tokenizer(
            prompt, 
            return_tensors="pt", 
            padding=True, 
            truncation=True,
            max_length=self.max_context_length - self.max_generation_length
        )
        
        # Generate response with controlled parameters
        with torch.no_grad():  # Disable gradients for inference
            outputs = self.generator_model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,  # Add attention mask
                max_length=inputs.input_ids.shape[1] + self.max_generation_length,
                num_return_sequences=1,      # Generate one response
                temperature=0.7,             # Control randomness (lower = more focused)
                do_sample=True,              # Enable sampling for variety
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                no_repeat_ngram_size=3      # Avoid repetitive phrases
            )
        
        # Decode generated response
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated answer (after "Answer: TCS")
        if "Answer: TCS" in generated_text:
            answer = "TCS" + generated_text.split("Answer: TCS")[-1].strip()
        elif "Answer:" in generated_text:
            answer = generated_text.split("Answer:")[-1].strip()
        else:
            # Fallback: get text after the prompt
            answer = generated_text[len(prompt):].strip()
        
        # Clean up the answer
        answer = answer.split('\n')[0]  # Take only first line
        if len(answer) > 200:  # Limit answer length
            answer = answer[:200] + "..."
            
        return answer if answer else "No answer generated."
    
    def complete_rag_pipeline(self, query: str, stage1_k: int = 15, final_k: int = 3) -> Dict[str, Any]:
        """
        Complete end-to-end RAG pipeline: retrieve + generate
        
        Args:
            query: User's question
            stage1_k: Number of candidates for Stage 1 retrieval
            final_k: Number of final chunks for generation
            
        Returns:
            Dictionary with query, retrieved chunks, and generated answer
        """
        print(f"\n{'='*70}")
        print(f"COMPLETE RAG PIPELINE: '{query}'")
        print(f"{'='*70}")
        
        # Step 1: Multi-stage retrieval
        print("\n--- STEP 1: RETRIEVAL ---")
        retrieved_chunks = self.advanced_rag.multi_stage_retrieve(
            query, stage1_k=stage1_k, final_k=final_k
        )
        
        # Step 2: Create prompt with retrieved context
        print(f"\n--- STEP 2: PROMPT CREATION ---")
        prompt = self.create_prompt_template(query, retrieved_chunks)
        
        # Step 3: Limit context to fit model window
        prompt = self.limit_context_tokens(prompt)
        print(f"Final prompt length: {len(self.tokenizer.encode(prompt))} tokens")
        
        # Step 4: Generate response
        print(f"\n--- STEP 3: RESPONSE GENERATION ---")
        generated_answer = self.generate_response(prompt)
        
        # Prepare final result
        result = {
            'query': query,
            'retrieved_chunks': retrieved_chunks,
            'generated_answer': generated_answer,
            'prompt_used': prompt
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
        
        print(f"\nGENERATED ANSWER:")
        print(f"{result['generated_answer']}")
    
    def interactive_rag_session(self):
        """
        Start an interactive RAG Q&A session
        """
        print(f"\n{'='*70}")
        print("INTERACTIVE RAG FINANCIAL Q&A SYSTEM")
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
        
        # Create complete RAG system
        rag_system = ResponseGenerator(indexer)
        
        # Test with sample questions
        test_questions = [
            "What was TCS revenue in 2025?",
            "How much did TCS spend on employees?",
            "What is TCS profit margin?"
        ]
        
        print("=== Testing Complete RAG System ===")
        
        for query in test_questions:
            result = rag_system.complete_rag_pipeline(query)
            rag_system.display_complete_result(result)
            print("\n" + "="*70)
        
        # Uncomment to start interactive session
        # rag_system.interactive_rag_session()
        
    except FileNotFoundError:
        print("Error: Index files not found.")
        print("Please run the full pipeline:")
        print("1. python data_processor.py")
        print("2. python embedding_indexer.py")
        print("3. python response_generator.py")
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure all required packages are installed with: pip install -r requirements.txt")