import streamlit as st
import time
import sys
import os
from typing import Dict, Any

# Add the current directory to Python path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from embedding_indexer import EmbeddingIndexer
from simple_response_generator import SimpleResponseGenerator
from guardrails import RAGGuardrails

class StreamlitRAGApp:
    """
    Streamlit web interface for the RAG Financial Q&A System
    """
    
    def __init__(self):
        """Initialize the RAG system components"""
        self.indexer = None
        self.rag_system = None
        self.guardrails = None
        self.system_initialized = False
    
    @st.cache_resource
    def initialize_system(_self):
        """
        Initialize RAG system components (cached for performance)
        """
        try:
            # Get correct paths
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(current_dir)
            indexes_path = os.path.join(project_root, "data", "indexes")
            
            # Initialize components
            indexer = EmbeddingIndexer("all-MiniLM-L6-v2")
            
            # Load pre-built indexes
            indexer.load_indexes(indexes_path)
            
            # Initialize RAG system and guardrails
            rag_system = SimpleResponseGenerator(indexer)
            guardrails = RAGGuardrails()
            
            return indexer, rag_system, guardrails, True
            
        except Exception as e:
            st.error(f"Error initializing system: {e}")
            return None, None, None, False
    
    def setup_page_config(self):
        """Configure Streamlit page settings"""
        st.set_page_config(
            page_title="TCS Financial Q&A RAG System",
            layout="wide"
        )
    
    def render_header(self):
        """Render the main header and description"""
        st.title("TCS Financial Q&A RAG System")
        st.markdown("---")
        
        st.markdown("""
        **Advanced Retrieval-Augmented Generation System for TCS Financial Data**
        
        This system uses multi-stage retrieval with guardrails for accurate financial Q&A.
        """)
        
        st.markdown("---")
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process user query through the complete RAG pipeline
        
        Args:
            query: User input query
            
        Returns:
            Dictionary with results and metadata
        """
        start_time = time.time()
        
        # Apply guardrails first
        temp_response = "Processing..."
        temp_chunks = []
        validation_result = self.guardrails.apply_guardrails(query, temp_response, temp_chunks)
        
        # Check input validation
        if not validation_result['input_validation']['is_valid']:
            processing_time = time.time() - start_time
            return {
                'success': False,
                'answer': self.guardrails.create_safe_response(validation_result),
                'confidence_score': 0.0,
                'method_used': 'Input Validation Filter',
                'response_time': processing_time,
                'validation_issues': validation_result['input_validation']['issues'],
                'retrieved_chunks': []
            }
        
        try:
            # Run RAG pipeline
            rag_result = self.rag_system.complete_rag_pipeline(query, stage1_k=15, final_k=3)
            
            # Apply output guardrails
            final_validation = self.guardrails.apply_guardrails(
                query, 
                rag_result['extracted_answer'], 
                rag_result['retrieved_chunks']
            )
            
            processing_time = time.time() - start_time
            
            # Determine final answer
            if final_validation['overall_valid']:
                final_answer = rag_result['extracted_answer']
                confidence = final_validation['final_confidence']
                method = "Context Extraction RAG"
            else:
                final_answer = self.guardrails.create_safe_response(final_validation)
                confidence = 0.2
                method = "Guardrail Filtered"
            
            return {
                'success': True,
                'answer': final_answer,
                'confidence_score': confidence,
                'method_used': method,
                'response_time': processing_time,
                'validation_issues': final_validation['all_issues'],
                'retrieved_chunks': rag_result['retrieved_chunks']
            }
            
        except Exception as e:
            processing_time = time.time() - start_time
            return {
                'success': False,
                'answer': f"Error processing query: {str(e)}",
                'confidence_score': 0.0,
                'method_used': 'Error Handler',
                'response_time': processing_time,
                'validation_issues': ['System error occurred'],
                'retrieved_chunks': []
            }
    
    def render_main_interface(self):
        """Render the main query interface"""
        
        # Query input section
        st.subheader("Ask Your Question")
        
        # Text area for query input
        user_query = st.text_area(
            "Enter your query about TCS financial data:",
            height=100,
            placeholder="Example: What was TCS revenue in 2025?"
        )
        
        # Ask button
        col1, col2 = st.columns([1, 4])
        with col1:
            ask_button = st.button("Ask", type="primary")
        
        # Process query when button is clicked
        if ask_button and user_query.strip():
            with st.spinner("Processing your question..."):
                result = self.process_query(user_query.strip())
            
            # Display results
            self.display_results(user_query, result)
        
        elif ask_button and not user_query.strip():
            st.warning("Please enter a question about TCS financial data.")
    
    def display_results(self, query: str, result: Dict[str, Any]):
        """
        Display the query results in a structured format
        
        Args:
            query: Original user query
            result: Processing result dictionary
        """
        st.markdown("---")
        st.subheader("Results")
        
        # Main answer section
        st.markdown("**Answer:**")
        if result['success'] and result['confidence_score'] > 0.5:
            st.success(result['answer'])
        elif result['success'] and result['confidence_score'] > 0.2:
            st.warning(result['answer'])
        else:
            st.error(result['answer'])
        
        # Metrics in columns
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Confidence Score", f"{result['confidence_score']:.2f}")
        
        with col2:
            st.metric("Method Used", result['method_used'])
        
        with col3:
            st.metric("Response Time", f"{result['response_time']:.2f}s")
        
        # Validation issues (if any)
        if result.get('validation_issues'):
            with st.expander("Validation Notices"):
                for issue in result['validation_issues']:
                    st.write(f"• {issue}")
        
        # Retrieved chunks details
        if result.get('retrieved_chunks'):
            with st.expander("Retrieved Context"):
                for i, chunk in enumerate(result['retrieved_chunks'], 1):
                    st.markdown(f"**Source {i}** (Score: {chunk.get('cross_encoder_score', 0):.4f})")
                    st.write(chunk['chunk_text'][:300] + "..." if len(chunk['chunk_text']) > 300 else chunk['chunk_text'])
                    st.markdown("---")
    
    def run(self):
        """Main application runner"""
        # Setup page
        self.setup_page_config()
        
        # Initialize system
        if not self.system_initialized:
            with st.spinner("Initializing RAG system..."):
                self.indexer, self.rag_system, self.guardrails, self.system_initialized = self.initialize_system()
        
        # Render UI
        self.render_header()
        
        # Check if system is ready
        if not self.system_initialized:
            st.error("System initialization failed. Please check that:")
            st.write("1. Run `python rag-system/run_full_pipeline.py` first from the project root")
            st.write("2. Install requirements: `pip install -r requirements.txt`")
            st.write("3. Ensure the TCS dataset exists in `data/tcs_qa_dataset.csv`")
            st.stop()
        
        # Render main interface
        self.render_main_interface()

# Main execution
if __name__ == "__main__":
    app = StreamlitRAGApp()
    app.run()