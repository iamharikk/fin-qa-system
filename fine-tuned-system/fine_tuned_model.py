import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
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
        self.qa_pipeline = None
        self.initialized = False
        
        print(f"Initializing fine-tuned DistilBERT model: {model_name}")
    
    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        try:
            # Load tokenizer and model from Hugging Face Hub
            print("Loading tokenizer...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            print("Loading DistilBERT model...")
            self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
            
            # Create question-answering pipeline
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1  # Use GPU if available
            )
            
            self.initialized = True
            print("Fine-tuned DistilBERT model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.initialized = False
    
    def create_context(self) -> str:
        """
        Create context from TCS financial data for the QA model
        Since DistilBERT QA needs context, we provide the financial data
        """
        context = """
        TCS Financial Data:
        TCS revenue in 2025 was Rs 21485300 crores. TCS revenue in 2024 was Rs 20235900 crores.
        TCS other income in 2025 was Rs 950700 crores. TCS other income in 2024 was Rs 626800 crores.
        TCS total income in 2025 was Rs 22436000 crores. TCS total income in 2024 was Rs 20862700 crores.
        TCS net profit in 2025 was Rs 4805700 crores. TCS net profit in 2024 was Rs 4355900 crores.
        TCS operating profit in 2025 was Rs 5792900 crores. TCS operating profit in 2024 was Rs 5584700 crores.
        TCS profit before tax in 2025 was Rs 6251300 crores. TCS profit before tax in 2024 was Rs 5755500 crores.
        TCS employee cost in 2025 was Rs 10730000 crores. TCS employee cost in 2024 was Rs 10313900 crores.
        TCS total expenses in 2025 were Rs 15692400 crores. TCS total expenses in 2024 were Rs 14651200 crores.
        TCS cash and bank balance in 2025 was Rs 715200 crores. TCS cash and bank balance in 2024 was Rs 659900 crores.
        TCS sundry debtors in 2025 were Rs 5176700 crores. TCS sundry debtors in 2024 were Rs 4606800 crores.
        TCS investments in 2025 were Rs 3280200 crores. TCS investments in 2024 were Rs 3224500 crores.
        TCS earnings per share in 2025 was Rs 13282. TCS earnings per share in 2024 was Rs 12039.
        TCS book value per share in 2025 was Rs 20900. TCS book value per share in 2024 was Rs 19933.
        TCS networth in 2025 was Rs 7561700 crores. TCS networth in 2024 was Rs 7212000 crores.
        """
        return context
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process user query using the fine-tuned DistilBERT QA model
        
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
            # Get context for the QA model
            context = self.create_context()
            
            # Use the fine-tuned DistilBERT QA model
            result = self.qa_pipeline(
                question=query,
                context=context,
                max_answer_len=150,
                handle_impossible_answer=True
            )
            
            # Extract answer and confidence
            answer = result['answer'].strip()
            confidence = result['score']
            
            # Post-process answer
            if not answer or confidence < 0.05:
                answer = "I couldn't find a confident answer in the TCS financial data."
                confidence = 0.0
            
            processing_time = time.time() - start_time
            
            return {
                'success': True,
                'answer': answer,
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