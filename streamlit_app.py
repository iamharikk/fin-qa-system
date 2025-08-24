import streamlit as st
import pandas as pd
import numpy as np
import time
import re
import sys
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any, List
from guardrails import validate_all_inputs
from output_guardrails import OutputGuardrails

# Add fine-tuned-system and rag-system to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'fine-tuned-system'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'rag-system'))

class SimpleRAGSystem:
    """
    Simple RAG system for cloud deployment using only TF-IDF
    """
    
    def __init__(self):
        """Initialize the RAG system"""
        self.dataset = None
        self.vectorizer = None
        self.document_vectors = None
        self.initialized = False
    
    def load_sample_data(self):
        """Load TCS financial data from CSV"""
        try:
            # Load data from CSV file
            csv_path = os.path.join(os.path.dirname(__file__), 'data', 'tcs_qa_dataset.csv')
            df = pd.read_csv(csv_path)
            
            # Convert to the expected format
            sample_data = []
            for _, row in df.iterrows():
                sample_data.append({
                    "question": row['Question'],
                    "answer": row['Answer']
                })
            
            self.dataset = pd.DataFrame(sample_data)
        except Exception as e:
            print(f"Error loading CSV data: {e}")
            # Fallback to hardcoded data if CSV fails
            sample_data = [
                {"question": "What was TCS revenue in 2025?", "answer": "TCS revenue in 2025 was Rs 21485300 crores."},
                {"question": "What was TCS revenue in 2024?", "answer": "TCS revenue in 2024 was Rs 20235900 crores."},
            ]
            self.dataset = pd.DataFrame(sample_data)
        
        # Create TF-IDF vectors
        combined_texts = []
        for _, row in self.dataset.iterrows():
            combined_text = f"{row['question']} {row['answer']}"
            combined_texts.append(combined_text)
        
        self.vectorizer = TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            ngram_range=(1, 2),
            max_features=1000
        )
        
        self.document_vectors = self.vectorizer.fit_transform(combined_texts)
        self.initialized = True
        
        return len(sample_data)
    
    def retrieve_relevant_info(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Retrieve relevant information using TF-IDF similarity"""
        if not self.initialized:
            return []
        
        # Transform query
        query_vector = self.vectorizer.transform([query])
        
        # Calculate similarities
        similarities = cosine_similarity(query_vector, self.document_vectors)[0]
        
        # Get top-k indices
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        # Return results with scores
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.05:  # Minimum similarity threshold
                results.append({
                    'question': self.dataset.iloc[idx]['question'],
                    'answer': self.dataset.iloc[idx]['answer'],
                    'similarity_score': similarities[idx],
                    'rank': len(results) + 1
                })
        
        return results
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """Process user query and return response"""
        start_time = time.time()
        
        # Basic input validation
        if not query or len(query.strip()) < 3:
            return {
                'success': False,
                'answer': "Please enter a valid question about TCS financial data.",
                'confidence_score': 0.0,
                'response_time': time.time() - start_time,
                'retrieved_info': []
            }
        
        # Retrieve relevant information
        retrieved_results = self.retrieve_relevant_info(query, top_k=3)
        
        # Extract answer
        if not retrieved_results:
            answer = "I don't have information to answer this question about TCS financial data."
            confidence = 0.0
        else:
            best_result = retrieved_results[0]
            if best_result['similarity_score'] > 0.2:
                answer = best_result['answer']
                confidence = best_result['similarity_score']
            else:
                answer = "I found some related information, but I'm not confident it directly answers your question."
                confidence = best_result['similarity_score']
        
        processing_time = time.time() - start_time
        
        return {
            'success': True,
            'answer': answer,
            'confidence_score': confidence,
            'response_time': processing_time,
            'retrieved_info': retrieved_results
        }

class FineTunedModel:
    """Fine-tuned model wrapper for Streamlit"""
    
    def __init__(self):
        self.model = None
        self.initialized = False
    
    def load_model(self):
        """Load the fine-tuned model"""
        try:
            from fine_tuned_model import FineTunedFinQA
            self.model = FineTunedFinQA()
            self.model.load_model()
            self.initialized = self.model.initialized
        except Exception as e:
            print(f"Error loading fine-tuned model: {e}")
            self.initialized = False
    
    def process_query(self, query: str) -> Dict[str, Any]:
        start_time = time.time()
        
        # Load model if not initialized
        if not self.initialized:
            self.load_model()
        
        # If still not initialized, return error
        if not self.initialized:
            return {
                'success': False,
                'answer': "Fine-tuned model could not be loaded. Using fallback response.",
                'confidence_score': 0.0,
                'response_time': time.time() - start_time,
                'retrieved_info': []
            }
        
        # Use the fine-tuned model
        result = self.model.process_query(query)
        
        return {
            'success': result['success'],
            'answer': result['answer'],
            'confidence_score': result['confidence_score'],
            'response_time': result['response_time'],
            'retrieved_info': []
        }

def main():
    """Main Streamlit application"""
    st.set_page_config(
        page_title="TCS Financial Q&A System",
        initial_sidebar_state="collapsed"
    )
    
    st.title("TCS Financial Q&A System")
    
    # Initialize systems
    if 'rag_system' not in st.session_state:
        st.session_state.rag_system = SimpleRAGSystem()
        st.session_state.rag_system.load_sample_data()
    
    if 'finetuned_system' not in st.session_state:
        st.session_state.finetuned_system = FineTunedModel()
    
    if 'output_guardrails' not in st.session_state:
        st.session_state.output_guardrails = OutputGuardrails()
    
    # Initialize Advanced RAG System
    if 'advanced_rag' not in st.session_state:
        try:
            from data_processor import DataProcessor
            from embedding_indexer import EmbeddingIndexer
            from simple_response_generator import SimpleResponseGenerator
            
            # Initialize with existing data
            data_processor = DataProcessor()
            csv_path = "data/tcs_qa_dataset.csv"
            
            if os.path.exists(csv_path):
                processed_data = data_processor.process_csv(csv_path)
                
                indexer = EmbeddingIndexer()
                chunks = []
                for item in processed_data:
                    chunks.append({
                        'text': f"Q: {item['question']} A: {item['answer']}",
                        'metadata': {'question': item['question'], 'answer': item['answer']}
                    })
                
                indexer.index_chunks(chunks)
                st.session_state.advanced_rag = SimpleResponseGenerator(indexer)
                st.success("Advanced RAG System initialized!")
            else:
                st.error("TCS dataset not found. Advanced RAG unavailable.")
                st.session_state.advanced_rag = None
                
        except Exception as e:
            st.error(f"Failed to initialize Advanced RAG: {e}")
            st.session_state.advanced_rag = None
    
    # Model selection radio buttons
    model_choice = st.radio(
        "Select Model:",
        ["Advanced RAG System", "Fine Tuned Model"],
        horizontal=True
    )
    
    # Query input text area
    user_query = st.text_area(
        "Enter your query about TCS financial data:",
        height=100,
        placeholder="Example: What was TCS revenue in 2025?"
    )
    
    # Ask button
    ask_button = st.button("Ask", type="primary")
    
    # Process query and show output
    if ask_button and user_query.strip():
        # Validate inputs using guardrails
        is_valid, error_message = validate_all_inputs(user_query.strip(), model_choice)
        
        if not is_valid:
            st.error(f"Input validation failed: {error_message}")
        else:
            with st.spinner("Processing..."):
                if model_choice == "Advanced RAG System":
                    if st.session_state.advanced_rag:
                        # Use advanced RAG system
                        advanced_result = st.session_state.advanced_rag.generate_response(
                            query=user_query.strip(),
                            broad_k=15,
                            final_k=5
                        )
                        result = {
                            'success': True,
                            'answer': advanced_result.get('generated_response', 'No response generated'),
                            'confidence_score': advanced_result.get('confidence_score', 0.5),
                            'response_time': advanced_result.get('total_time', 0.0)
                        }
                    else:
                        result = {
                            'success': False,
                            'answer': "Advanced RAG system is not available.",
                            'confidence_score': 0.0,
                            'response_time': 0.0
                        }
                else:
                    result = st.session_state.finetuned_system.process_query(user_query.strip())
            
            # Apply output guardrails validation
            if result['success']:
                validation = st.session_state.output_guardrails.validate_output(
                    query=user_query.strip(),
                    response=result['answer'],
                    confidence=result.get('confidence_score', 0.0)
                )
                
                # Update result based on validation
                if not validation['is_valid']:
                    result['answer'] = validation['corrected_response']
                    result['confidence_score'] = max(0.0, result['confidence_score'] * validation['validation_score'])
                
                # Show validation warnings for serious issues only (not format issues)
                serious_warnings = []
                for warning in validation['warnings']:
                    # Only show warnings for hallucination, factual inconsistency, and unrealistic values
                    if any(issue in warning.lower() for issue in [
                        'factual inconsistency', 'hallucination', 'unrealistic', 
                        'unsupported numbers', 'contradictory', 'future years'
                    ]):
                        serious_warnings.append(warning)
                
                if serious_warnings:
                    with st.expander("Output Validation Warnings"):
                        for warning in serious_warnings:
                            st.warning(warning)
            
            # Output section
            st.markdown("### Answer")
            if result['success'] and result['confidence_score'] > 0.3:
                st.success(result['answer'])
            elif result['success'] and result['confidence_score'] > 0.1:
                st.warning(result['answer'])
            else:
                st.error(result['answer'])
            
            # Metrics display
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence Score", f"{result['confidence_score']:.2f}")
            with col2:
                st.metric("Method Used", model_choice)
            with col3:
                st.metric("Response Time", f"{result['response_time']:.2f}s")
    
    elif ask_button and not user_query.strip():
        st.warning("Please enter a question about TCS financial data.")

if __name__ == "__main__":
    main()