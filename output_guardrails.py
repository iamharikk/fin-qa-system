import re
import json
from typing import Dict, Any, List, Tuple
from datetime import datetime

class OutputGuardrails:
    """
    Output-side guardrails to filter hallucinated or non-factual outputs
    Specifically designed for financial Q&A systems
    """
    
    def __init__(self):
        # Load reference financial data for fact-checking
        self.reference_data = self.load_reference_data()
        
        # Define financial patterns and ranges
        self.financial_patterns = {
            'currency': r'Rs\.?\s*(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*(?:crores?|lakhs?|millions?)?',
            'percentage': r'(\d+(?:\.\d+)?)\s*%',
            'year': r'(20\d{2})',
            'shares': r'(\d+(?:,\d{3})*(?:\.\d{2})?)\s*(?:lakh\s*)?shares?'
        }
        
        # Define reasonable ranges for TCS financial metrics
        self.metric_ranges = {
            'revenue': {'min': 100000, 'max': 500000},  # Rs crores
            'profit': {'min': 20000, 'max': 100000},    # Rs crores
            'expenses': {'min': 50000, 'max': 300000},  # Rs crores
            'share_capital': {'min': 300, 'max': 50000}, # Rs crores
            'earnings_per_share': {'min': 50, 'max': 200}, # Rs
            'book_value': {'min': 150, 'max': 300}      # Rs
        }
    
    def load_reference_data(self) -> Dict[str, Any]:
        """Load reference financial data for fact-checking"""
        try:
            # Try to load from CSV
            import pandas as pd
            df = pd.read_csv('data/tcs_qa_dataset.csv')
            
            reference = {}
            for _, row in df.iterrows():
                question = row['Question'].lower()
                answer = row['Answer']
                
                # Extract key-value pairs for fact-checking
                if 'revenue' in question or 'sales turnover' in question:
                    year = self.extract_year(question)
                    value = self.extract_currency_value(answer)
                    if year and value:
                        reference[f'revenue_{year}'] = value
                
                elif 'net profit' in question:
                    year = self.extract_year(question)
                    value = self.extract_currency_value(answer)
                    if year and value:
                        reference[f'net_profit_{year}'] = value
                
                elif 'share capital' in question:
                    year = self.extract_year(question)
                    value = self.extract_currency_value(answer)
                    if year and value:
                        reference[f'share_capital_{year}'] = value
                        
            return reference
            
        except Exception as e:
            print(f"Warning: Could not load reference data: {e}")
            return {}
    
    def extract_year(self, text: str) -> str:
        """Extract year from text"""
        match = re.search(self.financial_patterns['year'], text)
        return match.group(1) if match else None
    
    def extract_currency_value(self, text: str) -> float:
        """Extract currency value and convert to standard format (crores)"""
        match = re.search(self.financial_patterns['currency'], text.replace(',', ''))
        if not match:
            return None
            
        value = float(match.group(1))
        
        # Convert to crores based on unit
        if 'lakh' in text.lower():
            value = value / 100  # Convert lakhs to crores
        elif 'million' in text.lower():
            value = value / 10   # Convert millions to crores
            
        return value
    
    def validate_output(self, query: str, response: str, confidence: float = 0.0, 
                       retrieved_context: List[str] = None) -> Dict[str, Any]:
        """
        Comprehensive output validation for financial responses
        
        Returns:
            Dict with validation results and flags
        """
        validation_result = {
            'is_valid': True,
            'confidence_adjustment': 0.0,
            'flags': [],
            'warnings': [],
            'corrected_response': response,
            'validation_score': 1.0
        }
        
        # 1. Factual consistency check
        fact_check = self.check_factual_consistency(query, response)
        if not fact_check['is_consistent']:
            validation_result['flags'].append('factual_inconsistency')
            validation_result['warnings'].extend(fact_check['issues'])
            validation_result['confidence_adjustment'] -= 0.3
        
        # 2. Hallucination detection
        hallucination_check = self.detect_hallucination(response, retrieved_context)
        if hallucination_check['has_hallucination']:
            validation_result['flags'].append('potential_hallucination')
            validation_result['warnings'].extend(hallucination_check['reasons'])
            validation_result['confidence_adjustment'] -= 0.4
        
        # 3. Range validation for financial metrics
        range_check = self.validate_financial_ranges(response)
        if not range_check['in_range']:
            validation_result['flags'].append('unrealistic_values')
            validation_result['warnings'].extend(range_check['issues'])
            validation_result['confidence_adjustment'] -= 0.2
        
        # 4. Format consistency check
        format_check = self.check_response_format(response)
        if not format_check['is_proper']:
            validation_result['flags'].append('format_issues')
            validation_result['warnings'].extend(format_check['issues'])
            validation_result['confidence_adjustment'] -= 0.1
        
        # 5. Temporal consistency check
        temporal_check = self.check_temporal_consistency(query, response)
        if not temporal_check['is_consistent']:
            validation_result['flags'].append('temporal_inconsistency')
            validation_result['warnings'].extend(temporal_check['issues'])
            validation_result['confidence_adjustment'] -= 0.2
        
        # Calculate final validation score
        validation_result['validation_score'] = max(0.0, 1.0 + validation_result['confidence_adjustment'])
        
        # Determine if output should be blocked
        if validation_result['validation_score'] < 0.3:
            validation_result['is_valid'] = False
            validation_result['corrected_response'] = self.create_safe_fallback_response(query, validation_result['flags'])
        
        return validation_result
    
    def check_factual_consistency(self, query: str, response: str) -> Dict[str, Any]:
        """Check if response is factually consistent with reference data"""
        result = {'is_consistent': True, 'issues': []}
        
        # Extract financial claims from response
        claims = self.extract_financial_claims(response)
        
        for claim in claims:
            # Check against reference data
            if claim['type'] == 'revenue':
                ref_key = f"revenue_{claim['year']}"
                if ref_key in self.reference_data:
                    expected_value = self.reference_data[ref_key]
                    if abs(claim['value'] - expected_value) > expected_value * 0.1:  # 10% tolerance
                        result['is_consistent'] = False
                        result['issues'].append(f"Revenue for {claim['year']}: claimed {claim['value']}, expected ~{expected_value}")
            
            elif claim['type'] == 'net_profit':
                ref_key = f"net_profit_{claim['year']}"
                if ref_key in self.reference_data:
                    expected_value = self.reference_data[ref_key]
                    if abs(claim['value'] - expected_value) > expected_value * 0.1:
                        result['is_consistent'] = False
                        result['issues'].append(f"Net profit for {claim['year']}: claimed {claim['value']}, expected ~{expected_value}")
        
        return result
    
    def detect_hallucination(self, response: str, retrieved_context: List[str] = None) -> Dict[str, Any]:
        """Detect potential hallucinations in the response"""
        result = {'has_hallucination': False, 'reasons': []}
        
        # Check 1: Response contains information not in retrieved context
        if retrieved_context:
            response_numbers = set(re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', response.replace(',', '')))
            context_numbers = set()
            for ctx in retrieved_context:
                context_numbers.update(re.findall(r'\d+(?:,\d{3})*(?:\.\d+)?', ctx.replace(',', '')))
            
            unsupported_numbers = response_numbers - context_numbers
            if unsupported_numbers:
                result['has_hallucination'] = True
                result['reasons'].append(f"Contains unsupported numbers: {list(unsupported_numbers)[:3]}")
        
        # Check 2: Unrealistic financial statements
        if any(phrase in response.lower() for phrase in [
            'trillion', 'billion dollars', 'negative revenue', 'infinite profit'
        ]):
            result['has_hallucination'] = True
            result['reasons'].append("Contains unrealistic financial statements")
        
        # Check 3: Contradictory statements within response
        if self.has_internal_contradictions(response):
            result['has_hallucination'] = True
            result['reasons'].append("Contains contradictory statements")
        
        return result
    
    def validate_financial_ranges(self, response: str) -> Dict[str, Any]:
        """Validate that financial values are within realistic ranges"""
        result = {'in_range': True, 'issues': []}
        
        # Extract and validate currency values
        currency_matches = re.finditer(self.financial_patterns['currency'], response)
        
        for match in currency_matches:
            value = float(match.group(1).replace(',', ''))
            
            # Determine metric type from surrounding context
            context = response[max(0, match.start()-50):match.end()+50].lower()
            
            if any(word in context for word in ['revenue', 'sales', 'turnover']):
                if not (self.metric_ranges['revenue']['min'] <= value <= self.metric_ranges['revenue']['max']):
                    result['in_range'] = False
                    result['issues'].append(f"Revenue value {value} crores is outside realistic range")
            
            elif any(word in context for word in ['profit', 'earnings']):
                if not (self.metric_ranges['profit']['min'] <= value <= self.metric_ranges['profit']['max']):
                    result['in_range'] = False
                    result['issues'].append(f"Profit value {value} crores is outside realistic range")
            
            elif any(word in context for word in ['share capital']):
                if not (self.metric_ranges['share_capital']['min'] <= value <= self.metric_ranges['share_capital']['max']):
                    result['in_range'] = False
                    result['issues'].append(f"Share capital value {value} crores is outside realistic range")
        
        return result
    
    def check_response_format(self, response: str) -> Dict[str, Any]:
        """Check if response follows proper financial reporting format"""
        result = {'is_proper': True, 'issues': []}
        
        # Check for proper currency format
        if 'rs' in response.lower() and not re.search(r'Rs\.?\s*\d', response):
            result['is_proper'] = False
            result['issues'].append("Improper currency formatting")
        
        # Check for complete sentences
        if not response.strip().endswith('.') and len(response) > 10:
            result['is_proper'] = False
            result['issues'].append("Response should end with proper punctuation")
        
        # Check for TCS mention consistency
        if 'tcs' in response.lower() and any(word in response.lower() for word in ['infosys', 'wipro', 'cognizant']):
            result['is_proper'] = False
            result['issues'].append("Response mentions competitor companies")
        
        return result
    
    def check_temporal_consistency(self, query: str, response: str) -> Dict[str, Any]:
        """Check temporal consistency of the response"""
        result = {'is_consistent': True, 'issues': []}
        
        query_year = self.extract_year(query)
        response_years = re.findall(self.financial_patterns['year'], response)
        
        # If query asks for specific year, response should mention that year
        if query_year and query_year not in response_years:
            # Check if response has different year
            if response_years:
                result['is_consistent'] = False
                result['issues'].append(f"Query asks for {query_year} but response mentions {response_years}")
        
        # Check for future years (should not exist for historical financial data)
        current_year = datetime.now().year
        future_years = [year for year in response_years if int(year) > current_year]
        if future_years:
            result['is_consistent'] = False
            result['issues'].append(f"Response mentions future years: {future_years}")
        
        return result
    
    def extract_financial_claims(self, response: str) -> List[Dict[str, Any]]:
        """Extract structured financial claims from response"""
        claims = []
        
        # Extract revenue claims
        revenue_pattern = r'(?:revenue|sales|turnover).*?(?:in|for)?\s*(20\d{2}).*?Rs\.?\s*(\d+(?:,\d{3})*)'
        for match in re.finditer(revenue_pattern, response, re.IGNORECASE):
            claims.append({
                'type': 'revenue',
                'year': match.group(1),
                'value': float(match.group(2).replace(',', '')),
                'text': match.group(0)
            })
        
        # Extract profit claims
        profit_pattern = r'(?:net profit|profit).*?(?:in|for)?\s*(20\d{2}).*?Rs\.?\s*(\d+(?:,\d{3})*)'
        for match in re.finditer(profit_pattern, response, re.IGNORECASE):
            claims.append({
                'type': 'net_profit',
                'year': match.group(1),
                'value': float(match.group(2).replace(',', '')),
                'text': match.group(0)
            })
        
        return claims
    
    def has_internal_contradictions(self, response: str) -> bool:
        """Check for internal contradictions within the response"""
        # Extract all numerical claims
        claims = self.extract_financial_claims(response)
        
        # Check for contradictory claims about same metric and year
        seen_claims = {}
        for claim in claims:
            key = f"{claim['type']}_{claim['year']}"
            if key in seen_claims:
                # Check if values are significantly different (>5% difference)
                existing_value = seen_claims[key]
                if abs(claim['value'] - existing_value) > existing_value * 0.05:
                    return True
            else:
                seen_claims[key] = claim['value']
        
        return False
    
    def create_safe_fallback_response(self, query: str, flags: List[str]) -> str:
        """Create a safe fallback response when output fails validation"""
        
        if 'factual_inconsistency' in flags:
            return "I found some information about your query, but I cannot provide a confident answer due to potential factual inconsistencies in the available data. Please verify the information from official TCS financial reports."
        
        if 'potential_hallucination' in flags:
            return "I cannot provide a reliable answer to your query as the generated response contains information that may not be supported by the available data. Please refer to official TCS financial documents."
        
        if 'unrealistic_values' in flags:
            return "The financial values in the generated response appear unrealistic. Please check official TCS financial statements for accurate information."
        
        return "I cannot provide a confident answer to your financial query at this time. Please refer to official TCS financial reports for accurate information."
    
    def get_validation_summary(self, validation_result: Dict[str, Any]) -> str:
        """Get a human-readable summary of validation results"""
        if validation_result['is_valid']:
            return f"Output passed validation (Score: {validation_result['validation_score']:.2f})"
        else:
            flags_str = ", ".join(validation_result['flags'])
            return f"Output failed validation: {flags_str} (Score: {validation_result['validation_score']:.2f})"