import streamlit as st
import pandas as pd
import numpy as np
import time
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, Any, List

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
        """Load TCS financial data"""
        # TCS financial data
        sample_data = [
            {"question": "What was TCS revenue in 2025?", "answer": "TCS revenue in 2025 was Rs 21485300 crores."},
            {"question": "What was TCS revenue in 2024?", "answer": "TCS revenue in 2024 was Rs 20235900 crores."},
            {"question": "What was TCS other income in 2025?", "answer": "TCS other income in 2025 was Rs 950700 crores."},
            {"question": "What was TCS other income in 2024?", "answer": "TCS other income in 2024 was Rs 626800 crores."},
            {"question": "What was TCS net profit in 2025?", "answer": "TCS net profit in 2025 was Rs 4805700 crores."},
            {"question": "What was TCS net profit in 2024?", "answer": "TCS net profit in 2024 was Rs 4355900 crores."},
            {"question": "What was TCS employee cost in 2025?", "answer": "TCS employee cost in 2025 was Rs 10730000 crores."},
            {"question": "What was TCS employee cost in 2024?", "answer": "TCS employee cost in 2024 was Rs 10313900 crores."},
            {"question": "What were TCS total expenses in 2025?", "answer": "TCS total expenses in 2025 were Rs 15692400 crores."},
            {"question": "What were TCS total expenses in 2024?", "answer": "TCS total expenses in 2024 were Rs 14651200 crores."},
            {"question": "What was TCS cash and bank balance in 2025?", "answer": "TCS cash and bank balance in 2025 was Rs 715200 crores."},
            {"question": "What was TCS cash and bank balance in 2024?", "answer": "TCS cash and bank balance in 2024 was Rs 659900 crores."},
            {"question": "What was TCS operating profit in 2025?", "answer": "TCS operating profit in 2025 was Rs 5792900 crores."},
            {"question": "What was TCS operating profit in 2024?", "answer": "TCS operating profit in 2024 was Rs 5584700 crores."},
            {"question": "What was TCS profit before tax in 2025?", "answer": "TCS profit before tax in 2025 was Rs 6251300 crores."},
            {"question": "What was TCS profit before tax in 2024?", "answer": "TCS profit before tax in 2024 was Rs 5755500 crores."},
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
    """Placeholder for fine-tuned model"""
    
    def process_query(self, query: str) -> Dict[str, Any]:
        start_time = time.time()
        
        # Placeholder response for fine-tuned model
        answer = "Fine-tuned model response: This feature is under development. Currently showing placeholder results."
        
        return {
            'success': True,
            'answer': answer,
            'confidence_score': 0.85,
            'response_time': time.time() - start_time,
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
    
    # Model selection radio buttons
    model_choice = st.radio(
        "Select Model:",
        ["RAG System", "Fine Tuned Model"],
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
        with st.spinner("Processing..."):
            if model_choice == "RAG System":
                result = st.session_state.rag_system.process_query(user_query.strip())
            else:
                result = st.session_state.finetuned_system.process_query(user_query.strip())
        
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