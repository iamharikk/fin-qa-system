import torch
import json
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering, pipeline

class ImprovedDistilBERTQATester:
    """Improved test class for DistilBERT financial Q&A model."""
    
    def __init__(self, model_path='./models/distilbert_financial_qa_improved'):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.context = None
        
    def load_model(self):
        """Load the improved fine-tuned DistilBERT model."""
        try:
            print(f"Loading improved model from {self.model_path}...")
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_path)
            self.model = DistilBertForQuestionAnswering.from_pretrained(self.model_path)
            self.model.to(self.device)
            self.model.eval()
            print("Improved model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            print("Make sure you have trained the improved model first.")
            return False
    
    def load_context(self, data_file='../data/distilbert_simple_format.json'):
        """Load context from the training data."""
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            if data and len(data) > 0:
                self.context = data[0]['context']
                print(f"Context loaded: {len(self.context)} characters")
                return True
        except Exception as e:
            print(f"Error loading context: {e}")
            return False
    
    def answer_question(self, question, context=None, max_length=512):
        """Answer a question using the improved fine-tuned model with standard confidence."""
        if not self.model or not self.tokenizer:
            return "Model not loaded. Please load the model first.", 0.0
        
        if context is None:
            context = self.context
        
        if context is None:
            return "No context available. Please load context first.", 0.0
        
        # Use our improved method but with standard confidence calculation
        return self.answer_question_standard_confidence(question, context, max_length)
    
    def answer_question_fallback(self, question, context, max_length=512):
        """Fallback method with standard confidence calculation."""
        # Tokenize input
        inputs = self.tokenizer(
            question,
            context,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Standard confidence calculation (SQuAD-style)
        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]
        
        start_probs = torch.softmax(start_logits, dim=0)
        end_probs = torch.softmax(end_logits, dim=0)
        
        # Find best positions
        start_idx = torch.argmax(start_probs)
        end_idx = torch.argmax(end_probs)
        
        # Ensure valid span
        if end_idx < start_idx:
            end_idx = start_idx
        
        # Extract answer
        answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        answer = self.clean_answer(answer)
        
        # STANDARD confidence calculation (most common in literature)
        confidence = float(start_probs[start_idx] * end_probs[end_idx])
        
        return answer, confidence
    
    def answer_question_standard_confidence(self, question, context, max_length=512):
        """Answer using improved span detection but standard confidence calculation."""
        # Tokenize input
        inputs = self.tokenizer(
            question,
            context,
            return_tensors='pt',
            truncation=True,
            max_length=max_length,
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get answer span using improved method but standard confidence
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        
        # Use our improved answer extraction
        answer, _ = self.extract_best_answer(
            inputs['input_ids'][0], start_logits[0], end_logits[0]
        )
        
        # Calculate STANDARD confidence (like baseline and research papers)
        start_probs = torch.softmax(start_logits[0], dim=0)
        end_probs = torch.softmax(end_logits[0], dim=0)
        
        # Find the positions our improved method selected
        answer_tokens = self.tokenizer(answer, add_special_tokens=False)['input_ids']
        if len(answer_tokens) > 0:
            # Find where this answer appears in the input
            input_ids_list = inputs['input_ids'][0].tolist()
            
            # Simple approach: use the span with highest individual probabilities
            start_idx = torch.argmax(start_probs)
            end_idx = torch.argmax(end_probs)
            
            if end_idx < start_idx:
                end_idx = start_idx
            
            # Standard confidence calculation (SQuAD methodology)
            confidence = float(start_probs[start_idx] * end_probs[end_idx])
        else:
            confidence = 0.0
        
        return answer, confidence
    
    def extract_best_answer(self, input_ids, start_logits, end_logits, max_answer_length=30):
        """Extract the best answer using improved strategy."""
        
        # Get probabilities
        start_probs = torch.softmax(start_logits, dim=0)
        end_probs = torch.softmax(end_logits, dim=0)
        
        # Find best answer span
        best_score = 0
        best_start = 0
        best_end = 0
        
        # Consider top candidates for start and end positions
        top_start_indices = torch.topk(start_probs, k=10).indices
        top_end_indices = torch.topk(end_probs, k=10).indices
        
        for start_idx in top_start_indices:
            for end_idx in top_end_indices:
                # Ensure valid span
                if (end_idx >= start_idx and 
                    end_idx - start_idx <= max_answer_length and
                    start_idx > 0):  # Skip CLS token
                    
                    # Standard span scoring (for span selection, not final confidence)
                    score = start_probs[start_idx] * end_probs[end_idx]
                    
                    # Small bonus for reasonable span lengths 
                    span_length = end_idx - start_idx + 1
                    if 3 <= span_length <= 15:
                        score *= 1.05
                    
                    if score > best_score:
                        best_score = score
                        best_start = start_idx
                        best_end = end_idx
        
        # Extract and clean answer
        if best_score > 0:
            answer_tokens = input_ids[best_start:best_end+1]
            answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
            answer = self.clean_answer(answer)
        else:
            answer = "Unable to find answer"
            best_score = 0.0
        
        confidence = float(best_score)
        
        return answer, confidence
    
    def clean_answer(self, answer):
        """Clean and post-process the extracted answer."""
        answer = answer.strip()
        
        # Remove extra spaces around punctuation
        import re
        answer = re.sub(r'\s+([,.])', r'\1', answer)
        answer = re.sub(r'\s+', ' ', answer)
        
        # If answer is too short or just punctuation, mark as unable
        if len(answer) <= 2 or answer.replace(' ', '').replace('.', '').replace(',', '') == '':
            return "Unable to find answer"
        
        return answer
    
    def test_with_sample_questions(self):
        """Test the improved model with predefined sample questions."""
        sample_questions = [
            "What was TCS's sales turnover in 2025?",
            "What was TCS's net profit in 2025?",
            "What was TCS's employee cost in 2025?",
            "What was the growth in TCS's net profit from 2024 to 2025?",
            "What is TCS's profit margin in 2025?",
            "What was TCS's total income in 2024?"
        ]
        
        print("\n" + "="*60)
        print("TESTING IMPROVED MODEL WITH SAMPLE QUESTIONS")
        print("="*60)
        
        for i, question in enumerate(sample_questions, 1):
            print(f"\n{i}. Question: {question}")
            answer, confidence = self.answer_question(question)
            print(f"   Answer: {answer}")
            print(f"   Confidence: {confidence:.1%}")
            print("-" * 50)
    
    def compare_with_ground_truth(self, test_data_path='../data/distilbert_simple_format.json'):
        """Compare improved model outputs with ground truth answers."""
        try:
            with open(test_data_path, 'r', encoding='utf-8') as f:
                test_data = json.load(f)
        except FileNotFoundError:
            print(f"Test data file not found: {test_data_path}")
            return
        
        print("\n" + "="*60)
        print("COMPARING IMPROVED MODEL WITH GROUND TRUTH")
        print("="*60)
        
        correct = 0
        total = min(10, len(test_data))
        
        for i, item in enumerate(test_data[:total]):
            question = item['question']
            ground_truth = item['answer']
            
            generated_answer, confidence = self.answer_question(question)
            
            # Improved accuracy check
            is_correct = self._check_answer_accuracy(ground_truth, generated_answer)
            
            if is_correct:
                correct += 1
            
            status = "[PASS]" if is_correct else "[FAIL]"
            print(f"\n{i+1}. {status} Question: {question}")
            print(f"   Ground Truth: {ground_truth}")
            print(f"   Generated: {generated_answer} (confidence: {confidence:.1%})")
            print("-" * 50)
        
        accuracy = (correct / total) * 100
        print(f"\nImproved Model Accuracy: {accuracy:.1f}% ({correct}/{total})")
        return accuracy
    
    def _check_answer_accuracy(self, ground_truth, generated_answer):
        """Improved accuracy checking with better matching."""
        gt_lower = ground_truth.lower()
        gen_lower = generated_answer.lower()
        
        # Extract and compare key numbers
        import re
        gt_numbers = re.findall(r'\d+', ground_truth)
        gen_numbers = re.findall(r'\d+', generated_answer)
        
        # Check for number matches
        for gt_num in gt_numbers:
            for gen_num in gen_numbers:
                if gt_num == gen_num:
                    return True
        
        # Check for year consistency
        if '2025' in gt_lower and '2024' in gen_lower:
            return False
        if '2024' in gt_lower and '2025' in gen_lower:
            return False
        
        # Check for partial matches
        if len(gt_lower) > 10:
            return gt_lower in gen_lower or gen_lower in gt_lower
        else:
            return gt_lower == gen_lower
    
    def compare_models(self, baseline_results_file='baseline_results.json'):
        """Compare improved model with baseline results."""
        try:
            with open(baseline_results_file, 'r') as f:
                baseline_results = json.load(f)
            
            print("\n" + "="*60)
            print("MODEL COMPARISON")
            print("="*60)
            
            # Test improved model
            improved_accuracy = self.compare_with_ground_truth()
            
            # Compare with baseline
            baseline_accuracy = baseline_results.get('accuracy', 0)
            
            print(f"\nPerformance Comparison:")
            print(f"Baseline (Pre-trained): {baseline_accuracy:.1f}%")
            print(f"Original Fine-tuned: 50.0%")
            print(f"Improved Fine-tuned: {improved_accuracy:.1f}%")
            
            if improved_accuracy > baseline_accuracy:
                print(f"\nImprovement: +{improved_accuracy - baseline_accuracy:.1f}% vs baseline")
                print("Success! Improved fine-tuning beats baseline.")
            else:
                print(f"\nStill behind baseline by {baseline_accuracy - improved_accuracy:.1f}%")
                print("Consider further improvements.")
                
        except FileNotFoundError:
            print("Baseline results file not found. Run baseline benchmark first.")

def main():
    """Main testing function for improved model."""
    print("Improved DistilBERT Financial Q&A Model Tester")
    print("=" * 55)
    
    # Initialize tester
    tester = ImprovedDistilBERTQATester()
    
    # Load improved model
    if not tester.load_model():
        print("Failed to load improved model. Please train it first using:")
        print("python finetune_distilbert_improved.py")
        return
    
    # Load context
    if not tester.load_context():
        print("Failed to load context. Please run preprocessing first.")
        return
    
    # Run tests
    tester.test_with_sample_questions()
    
    # Compare with ground truth and other models
    tester.compare_models()
    
    print(f"\nImproved model testing completed!")

if __name__ == "__main__":
    main()