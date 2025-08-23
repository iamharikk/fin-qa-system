import re
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class RAGGuardrails:
    """
    Implementation of input-side and output-side guardrails for RAG system
    """
    
    def __init__(self):
        """Initialize guardrails with predefined rules and patterns"""
        
        # Financial domain keywords for relevance checking
        self.financial_keywords = [
            'revenue', 'profit', 'income', 'expense', 'cost', 'sales', 'turnover',
            'margin', 'earnings', 'dividend', 'cash', 'balance', 'asset', 'liability',
            'equity', 'share', 'investment', 'financial', 'quarter', 'year', 'growth',
            'tcs', 'company', 'business', 'performance', 'results', 'report'
        ]
        
        # Harmful/inappropriate content patterns
        self.harmful_patterns = [
            r'\b(hack|attack|breach|steal|fraud|illegal)\b',
            r'\b(personal|private|confidential|secret)\b.*\b(data|information)\b',
            r'\b(password|login|access|credential)\b',
        ]
        
        # Non-financial topic patterns
        self.irrelevant_patterns = [
            r'\b(weather|sports|entertainment|cooking|travel)\b',
            r'\b(movie|music|game|celebrity|fashion)\b',
            r'\b(health|medical|doctor|disease)\b',
        ]
        
        # Hallucination detection patterns
        self.hallucination_patterns = [
            r'I think|I believe|I guess|probably|maybe|might be',
            r'based on my knowledge|in my opinion|I assume',
            r'approximately|roughly|around|about \d+',
        ]
        
        # Initialize TF-IDF for domain relevance scoring
        self.domain_vectorizer = TfidfVectorizer(
            vocabulary=self.financial_keywords,
            lowercase=True,
            token_pattern=r'\b\w+\b'
        )
        
        # Create reference vector for financial domain
        financial_text = ' '.join(self.financial_keywords)
        self.financial_vector = self.domain_vectorizer.fit_transform([financial_text])
        
        print("Guardrails initialized with input and output validation")
    
    def validate_input_query(self, query: str) -> Dict[str, Any]:
        """
        Input-side validation: Check if query is appropriate and relevant
        
        Args:
            query: User input query
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'is_valid': True,
            'issues': [],
            'confidence_score': 1.0,
            'filtered_query': query.strip()
        }
        
        # Basic sanitization
        query_clean = query.strip().lower()
        
        # Check 1: Empty or too short query
        if len(query_clean) < 3:
            validation_result['is_valid'] = False
            validation_result['issues'].append("Query too short or empty")
            return validation_result
        
        # Check 2: Harmful content detection
        for pattern in self.harmful_patterns:
            if re.search(pattern, query_clean, re.IGNORECASE):
                validation_result['is_valid'] = False
                validation_result['issues'].append("Potentially harmful content detected")
                break
        
        # Check 3: Irrelevant topic detection
        for pattern in self.irrelevant_patterns:
            if re.search(pattern, query_clean, re.IGNORECASE):
                validation_result['is_valid'] = False
                validation_result['issues'].append("Query not related to financial domain")
                break
        
        # Check 4: Financial domain relevance scoring
        relevance_score = self._calculate_domain_relevance(query_clean)
        validation_result['confidence_score'] = relevance_score
        
        # Set threshold for relevance (0.1 is quite lenient)
        if relevance_score < 0.05:
            validation_result['issues'].append("Low relevance to financial domain")
            # Don't mark as invalid, just warn
        
        # Check 5: Basic SQL injection or code patterns
        suspicious_patterns = [r'select\s+.*from', r'drop\s+table', r'<script', r'javascript:']
        for pattern in suspicious_patterns:
            if re.search(pattern, query_clean, re.IGNORECASE):
                validation_result['is_valid'] = False
                validation_result['issues'].append("Suspicious code patterns detected")
                break
        
        return validation_result
    
    def validate_output_response(self, query: str, response: str, retrieved_chunks: List[Dict]) -> Dict[str, Any]:
        """
        Output-side validation: Check for hallucinations and factual consistency
        
        Args:
            query: Original user query
            response: Generated response
            retrieved_chunks: Context chunks used for generation
            
        Returns:
            Dictionary with validation results
        """
        validation_result = {
            'is_valid': True,
            'issues': [],
            'confidence_score': 1.0,
            'filtered_response': response
        }
        
        response_clean = response.strip().lower()
        
        # Check 1: Empty response
        if len(response_clean) < 5:
            validation_result['is_valid'] = False
            validation_result['issues'].append("Response too short or empty")
            return validation_result
        
        # Check 2: Hallucination pattern detection
        hallucination_count = 0
        for pattern in self.hallucination_patterns:
            matches = len(re.findall(pattern, response_clean, re.IGNORECASE))
            hallucination_count += matches
        
        if hallucination_count > 2:
            validation_result['issues'].append("High uncertainty language detected")
            validation_result['confidence_score'] *= 0.7
        
        # Check 3: Factual grounding in retrieved context
        context_relevance = self._check_context_grounding(response, retrieved_chunks)
        validation_result['confidence_score'] *= context_relevance
        
        if context_relevance < 0.3:
            validation_result['issues'].append("Response poorly grounded in retrieved context")
        
        # Check 4: Numerical consistency
        response_numbers = re.findall(r'\d+(?:\.\d+)?', response)
        context_numbers = []
        for chunk in retrieved_chunks:
            context_numbers.extend(re.findall(r'\d+(?:\.\d+)?', chunk.get('chunk_text', '')))
        
        # Check if response numbers appear in context
        grounded_numbers = sum(1 for num in response_numbers if num in context_numbers)
        if response_numbers and grounded_numbers / len(response_numbers) < 0.5:
            validation_result['issues'].append("Numerical values may not be grounded in context")
            validation_result['confidence_score'] *= 0.8
        
        # Check 5: Contradictory statements
        contradiction_patterns = [
            r'(increased|grew|rose).*but.*(decreased|fell|dropped)',
            r'(higher|more|greater).*but.*(lower|less|smaller)',
        ]
        for pattern in contradiction_patterns:
            if re.search(pattern, response_clean, re.IGNORECASE):
                validation_result['issues'].append("Potentially contradictory statements")
                validation_result['confidence_score'] *= 0.9
                break
        
        # Final confidence threshold
        if validation_result['confidence_score'] < 0.4:
            validation_result['issues'].append("Low overall confidence in response")
        
        return validation_result
    
    def _calculate_domain_relevance(self, query: str) -> float:
        """
        Calculate how relevant the query is to financial domain
        
        Args:
            query: Input query string
            
        Returns:
            Relevance score between 0 and 1
        """
        try:
            # Transform query using financial vocabulary
            query_vector = self.domain_vectorizer.transform([query])
            
            # Calculate similarity to financial domain
            similarity = cosine_similarity(query_vector, self.financial_vector)[0][0]
            
            # Also count direct keyword matches
            keyword_matches = sum(1 for keyword in self.financial_keywords if keyword in query.lower())
            keyword_score = min(keyword_matches / 5.0, 1.0)  # Normalize to max 1.0
            
            # Combined score
            relevance_score = max(similarity, keyword_score * 0.5)
            
            return relevance_score
            
        except Exception:
            # Fallback to keyword matching only
            keyword_matches = sum(1 for keyword in self.financial_keywords if keyword in query.lower())
            return min(keyword_matches / 3.0, 1.0)
    
    def _check_context_grounding(self, response: str, retrieved_chunks: List[Dict]) -> float:
        """
        Check how well the response is grounded in retrieved context
        
        Args:
            response: Generated response
            retrieved_chunks: Retrieved context chunks
            
        Returns:
            Grounding score between 0 and 1
        """
        if not retrieved_chunks:
            return 0.0
        
        # Combine all context text
        context_text = ' '.join([chunk.get('chunk_text', '') for chunk in retrieved_chunks])
        
        # Simple approach: check overlap of significant words
        response_words = set(re.findall(r'\b\w{4,}\b', response.lower()))  # Words with 4+ chars
        context_words = set(re.findall(r'\b\w{4,}\b', context_text.lower()))
        
        if not response_words:
            return 0.5  # Neutral if no significant words
        
        # Calculate word overlap
        overlap = len(response_words.intersection(context_words))
        grounding_score = overlap / len(response_words)
        
        return min(grounding_score, 1.0)
    
    def apply_guardrails(self, query: str, response: str, retrieved_chunks: List[Dict]) -> Dict[str, Any]:
        """
        Apply both input and output guardrails
        
        Args:
            query: User query
            response: Generated response
            retrieved_chunks: Retrieved context chunks
            
        Returns:
            Complete guardrail validation results
        """
        # Input validation
        input_validation = self.validate_input_query(query)
        
        # Output validation (only if input is valid)
        if input_validation['is_valid']:
            output_validation = self.validate_output_response(query, response, retrieved_chunks)
        else:
            output_validation = {'is_valid': False, 'issues': ['Input validation failed'], 'confidence_score': 0.0}
        
        # Combined result
        combined_result = {
            'overall_valid': input_validation['is_valid'] and output_validation['is_valid'],
            'input_validation': input_validation,
            'output_validation': output_validation,
            'final_confidence': min(input_validation['confidence_score'], output_validation['confidence_score']),
            'all_issues': input_validation['issues'] + output_validation['issues']
        }
        
        return combined_result
    
    def create_safe_response(self, validation_result: Dict[str, Any]) -> str:
        """
        Create a safe fallback response when validation fails
        
        Args:
            validation_result: Result from apply_guardrails
            
        Returns:
            Safe response string
        """
        if not validation_result['overall_valid']:
            if validation_result['input_validation']['issues']:
                return "I can only answer questions about TCS financial data. Please ask about revenue, profits, expenses, or other financial metrics."
            elif validation_result['output_validation']['issues']:
                return "I don't have enough reliable information to answer this question accurately. Please try rephrasing your question or ask about specific financial metrics."
        
        return "Please ask a question related to TCS financial performance."

# Example usage and testing
if __name__ == "__main__":
    # Initialize guardrails
    guardrails = RAGGuardrails()
    
    # Test cases
    test_cases = [
        {
            'query': 'What was TCS revenue in 2025?',
            'response': 'TCS revenue in 2025 was Rs 21485300 crores according to the financial data.',
            'chunks': [{'chunk_text': 'TCS revenue in 2025 was Rs 21485300 crores'}]
        },
        {
            'query': 'hack into TCS database',
            'response': 'I cannot help with that',
            'chunks': []
        },
        {
            'query': 'what is the weather today?',
            'response': 'It is sunny today',
            'chunks': []
        },
        {
            'query': 'TCS profit margins',
            'response': 'I think TCS probably has around 20% margins based on my knowledge',
            'chunks': [{'chunk_text': 'TCS financial performance data'}]
        }
    ]
    
    print("=== Testing Guardrails ===")
    
    for i, test in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: '{test['query']}'")
        result = guardrails.apply_guardrails(test['query'], test['response'], test['chunks'])
        
        print(f"Overall Valid: {result['overall_valid']}")
        print(f"Final Confidence: {result['final_confidence']:.3f}")
        if result['all_issues']:
            print(f"Issues: {', '.join(result['all_issues'])}")
        
        if not result['overall_valid']:
            safe_response = guardrails.create_safe_response(result)
            print(f"Safe Response: {safe_response}")
    
    print("\nGuardrails testing completed!")