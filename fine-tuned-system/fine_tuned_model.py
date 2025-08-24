import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import time
from typing import Dict, Any

class FineTunedFinQA:
    """
    Fine-tuned DistilBERT model for TCS Financial Q&A
    Uses the model from Hugging Face Hub: iamharikk/fin-qa-model1
    """
    
    def __init__(self, model_name: str = "iamharikk/fin-qa-model1"):
        """
        Initialize the fine-tuned model from Hugging Face
        
        Args:
            model_name: Hugging Face model identifier
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.generator = None
        self.initialized = False
        
        print(f"Initializing fine-tuned model: {model_name}")
    
    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        try:
            # Load tokenizer and model from Hugging Face Hub
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            print("Loading model...")
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            # Create text generation pipeline
            self.generator = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1,  # Use GPU if available
                max_length=512,
                do_sample=True,
                temperature=0.7
            )
            
            self.initialized = True
            print("Fine-tuned model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.initialized = False
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process user query using the fine-tuned model directly
        
        Args:
            query: User's financial question
            
        Returns:
            Dictionary with answer and metadata
        """
        start_time = time.time()
        
        # Check if model is initialized
        if not self.initialized:
            self.load_model()
            
        if not self.initialized:
            return {
                'success': False,
                'answer': "Fine-tuned model could not be loaded. Please try again later.",
                'confidence_score': 0.0,
                'response_time': time.time() - start_time,
                'model_used': self.model_name
            }
        
        # Basic input validation
        if not query or len(query.strip()) < 3:
            return {
                'success': False,
                'answer': "Please enter a valid question about TCS financial data.",
                'confidence_score': 0.0,
                'response_time': time.time() - start_time,
                'model_used': self.model_name
            }
        
        try:
            # Generate answer directly from the fine-tuned model
            # Format the input as the model was likely trained
            prompt = f"Question: {query}\nAnswer:"
            
            result = self.generator(
                prompt,
                max_new_tokens=100,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id,
                truncation=True
            )
            
            # Extract the generated answer
            generated_text = result[0]['generated_text']
            
            # Extract only the answer part (after "Answer:")
            if "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[-1].strip()
            else:
                answer = generated_text[len(prompt):].strip()
            
            # Clean up the answer
            answer = answer.split('\n')[0]  # Take only first line
            if len(answer) > 200:  # Limit answer length
                answer = answer[:200] + "..."
            
            # Simple confidence scoring based on answer quality
            confidence = 0.8 if answer and len(answer) > 10 else 0.3
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'answer': answer if answer else "No answer generated.",
                'confidence_score': confidence,
                'response_time': processing_time,
                'model_used': self.model_name
            }
            
        except Exception as e:
            print(f"Error processing query: {e}")
            return {
                'success': False,
                'answer': f"Error processing your question: {str(e)}",
                'confidence_score': 0.0,
                'response_time': time.time() - start_time,
                'model_used': self.model_name
            }

# Example usage and testing
if __name__ == "__main__":
    # Initialize the fine-tuned model
    finqa_model = FineTunedFinQA()
    finqa_model.load_model()
    
    # Test queries
    test_queries = [
        "What was TCS revenue in 2025?",
        "How much profit did TCS make in 2024?",
        "What are TCS employee costs?",
        "TCS cash and bank balance?"
    ]
    
    print("=== Testing Fine-Tuned Model ===")
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        result = finqa_model.process_query(query)
        print(f"Answer: {result['answer']}")
        print(f"Confidence: {result['confidence_score']:.3f}")
        print(f"Time: {result['response_time']:.2f}s")
        print("-" * 50)