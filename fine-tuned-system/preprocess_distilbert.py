import re
import json
from typing import List, Tuple, Dict

def parse_qa_pairs(file_path: str) -> List[Tuple[str, str]]:
    """Parse Q&A pairs from the text file."""
    qa_pairs = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract Q&A pairs using regex
    pattern = r'Question:\s*(.*?)\s*Answer:\s*(.*?)(?=\n\n|$)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    for question, answer in matches:
        question = question.strip()
        answer = answer.strip()
        if question and answer:
            qa_pairs.append((question, answer))
    
    return qa_pairs

def create_context_document(qa_pairs: List[Tuple[str, str]]) -> str:
    """Create a comprehensive context document from all Q&A pairs."""
    
    # Group financial data by categories
    context_parts = []
    
    # Add company introduction
    context_parts.append("TCS (Tata Consultancy Services) Financial Information:")
    
    # Process all Q&A pairs to create a coherent context
    financial_facts = []
    for question, answer in qa_pairs:
        # Create factual statements from Q&A pairs
        fact = f"{answer.replace('TCS', 'The company').replace('Rs.', 'Rupees')}"
        financial_facts.append(fact)
    
    # Join all facts into a coherent document
    context = "TCS Financial Data: " + " ".join(financial_facts)
    
    return context

def format_for_distilbert(qa_pairs: List[Tuple[str, str]]) -> Dict:
    """Format Q&A pairs for DistilBERT training with context."""
    
    # Create a single context document containing all financial information
    context = create_context_document(qa_pairs)
    
    training_data = {
        "context": context,
        "qas": []
    }
    
    for i, (question, answer) in enumerate(qa_pairs):
        # Find the answer in the context
        answer_start = context.lower().find(answer.lower())
        
        if answer_start == -1:
            # If exact answer not found, try to find key numbers
            numbers = re.findall(r'[\d,]+\.?\d*', answer)
            for num in numbers:
                answer_start = context.find(num)
                if answer_start != -1:
                    # Use the number as the answer
                    answer = num
                    break
        
        # If still not found, add the answer to context
        if answer_start == -1:
            context += f" {answer}"
            answer_start = len(context) - len(answer)
            training_data["context"] = context
        
        qa_item = {
            "id": f"tcs_financial_{i}",
            "question": question,
            "answers": [
                {
                    "text": answer,
                    "answer_start": answer_start
                }
            ]
        }
        
        training_data["qas"].append(qa_item)
    
    return training_data

def save_distilbert_data(data: Dict, output_path: str):
    """Save data in DistilBERT/SQuAD format."""
    
    squad_format = {
        "version": "2.0",
        "data": [
            {
                "title": "TCS Financial Information",
                "paragraphs": [data]
            }
        ]
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(squad_format, f, indent=2, ensure_ascii=False)

def create_simple_qa_format(qa_pairs: List[Tuple[str, str]]) -> List[Dict]:
    """Create simple question-answer format for easier training."""
    
    # Create a comprehensive context from all answers
    context_parts = []
    for question, answer in qa_pairs:
        context_parts.append(f"{answer}")
    
    context = " ".join(context_parts)
    
    simple_data = []
    for question, answer in qa_pairs:
        simple_data.append({
            "question": question,
            "context": context,
            "answer": answer
        })
    
    return simple_data

def main():
    input_file = "../data/q-and-a/fin-data-qa-pairs.txt"
    simple_output = "../data/distilbert_simple_format.json"
    
    print("Parsing Q&A pairs...")
    qa_pairs = parse_qa_pairs(input_file)
    print(f"Found {len(qa_pairs)} Q&A pairs")
    
    print("Creating simple Q&A format...")
    simple_data = create_simple_qa_format(qa_pairs)
    with open(simple_output, 'w', encoding='utf-8') as f:
        json.dump(simple_data, f, indent=2, ensure_ascii=False)
    print(f"Simple format data saved to {simple_output}")
    
    print("\nSample format:")
    print(f"Question: {simple_data[0]['question']}")
    print(f"Answer: {simple_data[0]['answer']}")
    print(f"Context length: {len(simple_data[0]['context'])} characters")

if __name__ == "__main__":
    main()