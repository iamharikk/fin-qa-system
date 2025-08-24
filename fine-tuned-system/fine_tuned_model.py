import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import time
import json
import os
from typing import Dict, Any

class FineTunedFinQA:
    """
    Fine-tuned DistilBERT model for TCS Financial Q&A
    Uses the model from Hugging Face Hub: iamharikk/fin-qa-model1
    """
    
    def __init__(self, model_name: str = "iamharikk/fin-qa-model1", use_local_model: bool = False, local_model_path: str = "./models/distilbert_financial_qa_improved"):
        """
        Initialize the fine-tuned model
        
        Args:
            model_name: Hugging Face model identifier (for external model)
            use_local_model: Whether to use locally trained model
            local_model_path: Path to local model directory
        """
        self.model_name = model_name
        self.use_local_model = use_local_model
        self.local_model_path = local_model_path
        self.tokenizer = None
        self.model = None
        self.qa_pipeline = None
        self.initialized = False
        self.context = None
        
        model_type = "local" if use_local_model else "Hugging Face"
        model_path = local_model_path if use_local_model else model_name
        print(f"Initializing {model_type} DistilBERT model: {model_path}")
    
    def load_model(self):
        """Load the fine-tuned model and tokenizer"""
        try:
            if self.use_local_model:
                # Load local model
                print("Loading local tokenizer...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.local_model_path)
                
                print("Loading local DistilBERT model...")
                self.model = AutoModelForQuestionAnswering.from_pretrained(self.local_model_path)
            else:
                # Load from Hugging Face Hub
                print("Loading tokenizer from Hugging Face...")
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                
                print("Loading DistilBERT model from Hugging Face...")
                self.model = AutoModelForQuestionAnswering.from_pretrained(self.model_name)
            
            # Create question-answering pipeline
            self.qa_pipeline = pipeline(
                "question-answering",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if torch.cuda.is_available() else -1  # Use GPU if available
            )
            
            # Load context from JSON if available, otherwise use hardcoded
            self.load_context()
            
            self.initialized = True
            model_type = "local" if self.use_local_model else "Hugging Face"
            print(f"{model_type} DistilBERT model loaded successfully!")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            self.initialized = False
    
    def load_context(self):
        """Load context from JSON file or use hardcoded fallback"""
        try:
            # Try to load context from JSON file
            json_path = os.path.join(os.path.dirname(__file__), '../data/distilbert_simple_format.json')
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if data and len(data) > 0:
                self.context = data[0]['context']
                print(f"Context loaded from JSON: {len(self.context)} characters")
                return
        except Exception as e:
            print(f"Could not load context from JSON: {e}")
        
        # Fallback to updated hardcoded context with correct values from CSV
        self.context = """
        TCS Financial Data:
        TCS sales turnover in 2025 was Rs 214853 crores. TCS sales turnover in 2024 was Rs 202359 crores.
        TCS total income in 2025 was Rs 224360 crores. TCS total income in 2024 was Rs 208627 crores.
        TCS other income in 2025 was Rs 9507 crores. TCS other income in 2024 was Rs 6268 crores.
        TCS reported net profit in 2025 was Rs 48057 crores. TCS reported net profit in 2024 was Rs 43559 crores.
        TCS operating profit in 2025 was Rs 57929 crores. TCS operating profit in 2024 was Rs 55847 crores.
        TCS profit before tax in 2025 was Rs 62513 crores. TCS profit before tax in 2024 was Rs 57555 crores.
        TCS employee cost in 2025 was Rs 107300 crores. TCS employee cost in 2024 was Rs 103139 crores.
        TCS total expenses in 2025 were Rs 156924 crores. TCS total expenses in 2024 were Rs 146512 crores.
        TCS earnings per share in 2025 was Rs 132. TCS earnings per share in 2024 was Rs 120.
        TCS book value per share in 2025 was Rs 209. TCS book value per share in 2024 was Rs 199.
        TCS cash and bank balance in 2025 was Rs 7152 crores. TCS cash and bank balance in 2024 was Rs 6599 crores.
        TCS sundry debtors in 2025 were Rs 51767 crores. TCS sundry debtors in 2024 were Rs 46068 crores.
        TCS investments in 2025 were Rs 32802 crores. TCS investments in 2024 were Rs 32245 crores.
        TCS networth in 2025 was Rs 75617 crores. TCS networth in 2024 was Rs 72120 crores.
        """
        print("Using fallback hardcoded context with correct values")
    
    def create_context(self) -> str:
        """Return the loaded context"""
        return self.context if self.context else ""
    
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