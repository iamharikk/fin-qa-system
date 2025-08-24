import torch
import json
import numpy as np
from transformers import (
    DistilBertTokenizerFast, 
    DistilBertForQuestionAnswering,
    Trainer,
    TrainingArguments,
    default_data_collator
)
from torch.utils.data import Dataset
import os
from sklearn.metrics import accuracy_score
import random

class ImprovedDistilBERTQADataset(Dataset):
    """Improved dataset with better answer position detection and data augmentation."""
    
    def __init__(self, tokenizer, data_file, max_length=512, augment_data=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        self.augment_data = augment_data
        
        # Load simple format data
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded {len(data)} Q&A examples")
        
        # Process each example with improved answer finding
        for item in data:
            question = item['question']
            context = item['context']
            answer = item['answer']
            
            # Clean answer for better matching
            clean_answer = self.clean_answer_text(answer)
            
            # Find answer in context with improved matching
            start_idx, end_idx = self.find_answer_in_context(context, clean_answer, answer)
            
            if start_idx != -1 and end_idx != -1:
                self.examples.append({
                    'question': question,
                    'context': context,
                    'answer': clean_answer,
                    'start_char': start_idx,
                    'end_char': end_idx
                })
                
                # Data augmentation: create variations
                if self.augment_data:
                    self.add_augmented_examples(question, context, clean_answer, start_idx, end_idx)
        
        print(f"Created {len(self.examples)} training examples (including augmentation)")
    
    def clean_answer_text(self, answer):
        """Extract the core answer from full sentence answers."""
        # Extract numerical values and key terms
        import re
        
        # Pattern for "TCS's X was Rs. Y crores"
        money_pattern = r'Rs\.\s*(\d+)\s*crores'
        money_match = re.search(money_pattern, answer)
        if money_match:
            return f"Rs. {money_match.group(1)} crores"
        
        # Pattern for percentages
        percent_pattern = r'(\d+)%'
        percent_match = re.search(percent_pattern, answer)
        if percent_match:
            return f"{percent_match.group(1)}%"
        
        # Pattern for share counts
        share_pattern = r'(\d+)\s*lakh\s*shares'
        share_match = re.search(share_pattern, answer)
        if share_match:
            return f"{share_match.group(1)} lakh shares"
        
        # Default: return the answer as is
        return answer
    
    def find_answer_in_context(self, context, clean_answer, original_answer):
        """Improved answer finding with multiple strategies."""
        
        # Strategy 1: Direct match of cleaned answer
        start_idx = context.find(clean_answer)
        if start_idx != -1:
            return start_idx, start_idx + len(clean_answer)
        
        # Strategy 2: Match original answer
        start_idx = context.find(original_answer)
        if start_idx != -1:
            return start_idx, start_idx + len(original_answer)
        
        # Strategy 3: Find key numbers
        import re
        numbers = re.findall(r'\d+', clean_answer)
        for num in numbers:
            # Look for the number in context
            pattern = rf'Rs\.\s*{num}\s*crores|{num}%|{num}\s*lakh'
            match = re.search(pattern, context)
            if match:
                return match.start(), match.end()
        
        # Strategy 4: Fuzzy matching for year-specific content
        if '2025' in clean_answer or '2024' in clean_answer:
            year = '2025' if '2025' in clean_answer else '2024'
            # Find content related to this year
            year_pattern = rf'in {year} was ([^.]+)'
            match = re.search(year_pattern, context)
            if match:
                return match.start(1), match.end(1)
        
        print(f"Warning: Could not find answer '{clean_answer}' in context")
        return -1, -1
    
    def add_augmented_examples(self, question, context, answer, start_idx, end_idx):
        """Add augmented examples to improve learning."""
        
        # Augmentation 1: Slightly different question phrasings
        question_variants = [
            question.replace("What was", "What is"),
            question.replace("How much was", "What was"),
            question.replace("TCS's", "TCS"),
        ]
        
        for variant in question_variants:
            if variant != question and len(variant) > 10:
                self.examples.append({
                    'question': variant,
                    'context': context,
                    'answer': answer,
                    'start_char': start_idx,
                    'end_char': end_idx
                })
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        example = self.examples[idx]
        
        # Tokenize with offset mapping for precise alignment
        encoding = self.tokenizer(
            example['question'],
            example['context'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt',
            return_offsets_mapping=True,
            verbose=False
        )
        
        # Find token positions using character positions
        offset_mapping = encoding['offset_mapping'].squeeze().tolist()
        start_char = example['start_char']
        end_char = example['end_char']
        
        start_token = 0
        end_token = 0
        
        # Improved token alignment
        for i, (offset_start, offset_end) in enumerate(offset_mapping):
            if offset_start == 0 and offset_end == 0:
                continue
                
            # Find start token with better precision
            if start_token == 0 and offset_start <= start_char < offset_end:
                start_token = i
            elif start_token == 0 and abs(offset_start - start_char) <= 2:  # Allow small offset
                start_token = i
            
            # Find end token with better precision
            if offset_start < end_char <= offset_end:
                end_token = i
                break
            elif abs(offset_end - end_char) <= 2:  # Allow small offset
                end_token = i
        
        # Ensure valid token positions
        if start_token == 0 or end_token == 0:
            # Fallback: use reasonable default positions
            start_token = max(1, min(50, len(offset_mapping) // 4))
            end_token = min(start_token + 10, self.max_length - 2)
        
        # Ensure end >= start
        if end_token < start_token:
            end_token = start_token + 1
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'start_positions': torch.tensor(start_token, dtype=torch.long),
            'end_positions': torch.tensor(end_token, dtype=torch.long)
        }

class ImprovedDistilBERTFinancialQA:
    """Improved DistilBERT fine-tuner with better training strategies."""
    
    def __init__(self, model_name='distilbert-base-cased-distilled-squad', output_dir='./models/distilbert_financial_qa_improved'):
        self.model_name = model_name
        self.output_dir = output_dir
        self.tokenizer = None
        self.model = None
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
    
    def setup_model_and_tokenizer(self):
        """Initialize tokenizer and model."""
        print("Loading DistilBERT tokenizer and model...")
        
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(self.model_name)
        self.model = DistilBertForQuestionAnswering.from_pretrained(self.model_name)
        
        print(f"Model loaded: {self.model_name}")
    
    def train_improved(self, data_file, epochs=8, batch_size=4, learning_rate=1e-5, 
                      warmup_ratio=0.1, weight_decay=0.01):
        """Improved training with better hyperparameters and strategies."""
        
        if not self.tokenizer or not self.model:
            self.setup_model_and_tokenizer()
        
        print("Preparing improved dataset...")
        dataset = ImprovedDistilBERTQADataset(self.tokenizer, data_file, augment_data=True)
        
        if len(dataset) == 0:
            print("No valid examples found in dataset!")
            return
        
        # Improved train/validation split
        train_size = int(0.85 * len(dataset))  # Use more data for training
        val_size = len(dataset) - train_size
        
        if val_size == 0:
            val_size = max(1, len(dataset) // 10)
            train_size = len(dataset) - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            dataset, [train_size, val_size], 
            generator=torch.Generator().manual_seed(42)  # Reproducible split
        )
        
        print(f"Training samples: {len(train_dataset)}")
        print(f"Validation samples: {len(val_dataset)}")
        
        # Calculate warmup steps
        total_steps = len(train_dataset) * epochs // batch_size
        warmup_steps = int(total_steps * warmup_ratio)
        
        # Improved training arguments
        training_args = TrainingArguments(
            output_dir=self.output_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=2,  # Effective batch size = batch_size * 2
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            warmup_steps=warmup_steps,
            
            # Improved evaluation and saving
            eval_strategy="steps",
            eval_steps=50,
            save_steps=100,
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            
            # Regularization and optimization
            dataloader_drop_last=False,
            dataloader_num_workers=0,
            remove_unused_columns=True,
            label_smoothing_factor=0.1,  # Slight label smoothing
            
            # Logging
            logging_dir=f'{self.output_dir}/logs',
            logging_steps=25,
            logging_first_step=True,
            
            # Mixed precision
            fp16=torch.cuda.is_available(),
            
            # Reproducibility
            seed=42,
            data_seed=42,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=default_data_collator,
            tokenizer=self.tokenizer,
        )
        
        print("Starting improved training...")
        print(f"Total training steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")
        print(f"Effective batch size: {batch_size * training_args.gradient_accumulation_steps}")
        
        # Train model
        trainer.train()
        
        # Save the fine-tuned model
        print(f"Saving improved model to {self.output_dir}...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        print("Improved training completed!")
        
        # Save training metrics
        logs = trainer.state.log_history
        with open(f"{self.output_dir}/training_metrics.json", 'w') as f:
            json.dump(logs, f, indent=2)

def main():
    """Main training function with improved parameters."""
    print("Improved DistilBERT Financial Q&A Fine-tuning")
    print("=" * 60)
    
    # Initialize improved trainer
    trainer = ImprovedDistilBERTFinancialQA()
    
    # Improved training parameters
    trainer.train_improved(
        data_file='../data/distilbert_simple_format.json',
        epochs=8,           # More epochs
        batch_size=4,       # Smaller batch size for stability
        learning_rate=1e-5, # Lower learning rate
        warmup_ratio=0.1,   # Gradual warmup
        weight_decay=0.01   # Regularization
    )
    
    print("\nImproved fine-tuning completed!")
    print("Test the model with: python test_distilbert_improved.py")

if __name__ == "__main__":
    main()