import torch
import json
import time
import re
from transformers import DistilBertTokenizerFast, DistilBertForQuestionAnswering
from statistics import mean

class BaselineBenchmark:
    """Benchmark pre-trained DistilBERT model before fine-tuning."""
    
    def __init__(self, model_name='distilbert-base-cased-distilled-squad'):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.context = None
        
    def load_pretrained_model(self):
        """Load the pre-trained DistilBERT model (SQuAD fine-tuned)."""
        try:
            print(f"Loading pre-trained model: {self.model_name}")
            print(f"Device: {self.device}")
            
            self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_name)
            self.model = DistilBertForQuestionAnswering.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            
            print("Pre-trained model loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def load_test_context(self, data_file='../data/distilbert_simple_format.json'):
        """Load context from the test data."""
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
        """Answer a question using the pre-trained model."""
        if not self.model or not self.tokenizer:
            return "Model not loaded", 0.0, 0.0
        
        if context is None:
            context = self.context
        
        if context is None:
            return "No context available", 0.0, 0.0
        
        # Record inference time
        start_time = time.time()
        
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
        
        # Get answer span
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits
        
        # Find best start and end positions
        start_idx = torch.argmax(start_logits, dim=1)[0]
        end_idx = torch.argmax(end_logits, dim=1)[0]
        
        # Ensure end >= start
        if end_idx < start_idx:
            end_idx = start_idx
        
        # Extract answer tokens
        answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        # Calculate inference time
        inference_time = time.time() - start_time
        
        # Calculate confidence
        start_confidence = torch.softmax(start_logits, dim=1)[0][start_idx].item()
        end_confidence = torch.softmax(end_logits, dim=1)[0][end_idx].item()
        confidence = (start_confidence + end_confidence) / 2
        
        # Clean up answer
        answer = answer.strip()
        if not answer:
            answer = "Unable to find answer"
            confidence = 0.0
        
        return answer, confidence, inference_time
    
    def load_test_questions(self, data_file='../data/distilbert_simple_format.json'):
        """Load test questions and ground truth answers."""
        try:
            with open(data_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return data
        except Exception as e:
            print(f"Error loading test data: {e}")
            return []
    
    def check_answer_accuracy(self, ground_truth, generated_answer):
        """Check if generated answer matches ground truth."""
        # Convert to lowercase for comparison
        gt_lower = ground_truth.lower()
        gen_lower = generated_answer.lower()
        
        # Extract numbers from both answers
        gt_numbers = re.findall(r'[\d,]+\.?\d*', ground_truth)
        gen_numbers = re.findall(r'[\d,]+\.?\d*', generated_answer)
        
        # Check if key numbers match
        for gt_num in gt_numbers:
            clean_gt = gt_num.replace(',', '')
            for gen_num in gen_numbers:
                clean_gen = gen_num.replace(',', '')
                if clean_gt == clean_gen:
                    return True
        
        # Check for partial string match (for non-numeric answers)
        if len(gt_lower) > 10:
            return gt_lower in gen_lower or gen_lower in gt_lower
        else:
            return gt_lower == gen_lower
    
    def run_baseline_benchmark(self, num_questions=10):
        """Run comprehensive baseline benchmark."""
        print("\n" + "="*80)
        print("BASELINE BENCHMARK - PRE-TRAINED DISTILBERT MODEL")
        print("="*80)
        
        # Load test data
        test_data = self.load_test_questions()
        if not test_data:
            print("No test data available")
            return None
        
        # Limit to specified number of questions
        test_data = test_data[:num_questions]
        
        results = {
            'total_questions': len(test_data),
            'correct_answers': 0,
            'accuracies': [],
            'confidences': [],
            'inference_times': [],
            'detailed_results': []
        }
        
        print(f"Testing {len(test_data)} questions...")
        print("-" * 80)
        
        for i, item in enumerate(test_data, 1):
            question = item['question']
            ground_truth = item['answer']
            
            # Get model prediction
            answer, confidence, inference_time = self.answer_question(question)
            
            # Check accuracy
            is_correct = self.check_answer_accuracy(ground_truth, answer)
            if is_correct:
                results['correct_answers'] += 1
            
            # Store metrics
            results['confidences'].append(confidence)
            results['inference_times'].append(inference_time)
            
            # Store detailed result
            result_detail = {
                'question': question,
                'ground_truth': ground_truth,
                'generated_answer': answer,
                'confidence': confidence,
                'inference_time': inference_time,
                'correct': is_correct
            }
            results['detailed_results'].append(result_detail)
            
            # Display result
            status = "CORRECT" if is_correct else "INCORRECT"
            print(f"{i:2d}. [{status}]")
            print(f"    Question: {question}")
            print(f"    Expected: {ground_truth}")
            print(f"    Generated: {answer}")
            print(f"    Confidence: {confidence:.1%} | Time: {inference_time:.3f}s")
            print("-" * 80)
        
        # Calculate final metrics
        accuracy = (results['correct_answers'] / results['total_questions']) * 100
        avg_confidence = mean(results['confidences']) * 100
        avg_inference_time = mean(results['inference_times'])
        
        results['accuracy'] = accuracy
        results['avg_confidence'] = avg_confidence
        results['avg_inference_time'] = avg_inference_time
        
        # Print summary
        print("\n" + "="*80)
        print("BASELINE BENCHMARK RESULTS")
        print("="*80)
        print(f"Overall Accuracy: {accuracy:.1f}% ({results['correct_answers']}/{results['total_questions']})")
        print(f"Average Confidence: {avg_confidence:.1f}%")
        print(f"Average Inference Time: {avg_inference_time:.3f} seconds")
        print(f"Total Time: {sum(results['inference_times']):.2f} seconds")
        
        # Performance categorization
        if accuracy >= 80:
            print("Excellent baseline performance!")
        elif accuracy >= 60:
            print("Good baseline performance")
        elif accuracy >= 40:
            print("Moderate baseline performance")
        else:
            print("Poor baseline performance - fine-tuning needed")
        
        print("="*80)
        
        return results
    
    def save_baseline_results(self, results, filename='baseline_results.json'):
        """Save benchmark results to JSON file."""
        if results:
            try:
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(results, f, indent=2, ensure_ascii=False)
                print(f"Results saved to {filename}")
            except Exception as e:
                print(f"Error saving results: {e}")

def main():
    """Main benchmarking function."""
    print("DistilBERT Baseline Benchmark")
    print("Testing pre-trained model performance on financial Q&A")
    
    # Initialize benchmark
    benchmark = BaselineBenchmark()
    
    # Load pre-trained model
    if not benchmark.load_pretrained_model():
        print("Failed to load model. Exiting...")
        return
    
    # Load context
    if not benchmark.load_test_context():
        print("Failed to load context. Please ensure data file exists.")
        return
    
    # Run benchmark
    results = benchmark.run_baseline_benchmark(num_questions=10)
    
    # Save results
    if results:
        benchmark.save_baseline_results(results)
        
        # Additional analysis
        print("\nDETAILED ANALYSIS:")
        print("-" * 50)
        
        correct_confidences = [r['confidence'] for r in results['detailed_results'] if r['correct']]
        incorrect_confidences = [r['confidence'] for r in results['detailed_results'] if not r['correct']]
        
        if correct_confidences:
            print(f"Avg confidence (correct answers): {mean(correct_confidences):.1%}")
        if incorrect_confidences:
            print(f"Avg confidence (incorrect answers): {mean(incorrect_confidences):.1%}")
        
        fast_responses = [t for t in results['inference_times'] if t < 0.1]
        slow_responses = [t for t in results['inference_times'] if t >= 0.1]
        
        print(f"Fast responses (<0.1s): {len(fast_responses)}")
        print(f"Slow responses (>=0.1s): {len(slow_responses)}")

if __name__ == "__main__":
    main()