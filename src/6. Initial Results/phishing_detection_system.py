#!/usr/bin/env python3
"""
Phishing Detection System using BERT
=====================================

This script implements a comprehensive phishing detection system using BERT/DistilBERT
for text classification. It processes 6 different email datasets and trains a model
optimized for Chrome Extension deployment.

Author: AI Assistant
Date: 2024
"""

import pandas as pd
import numpy as np
import re
import warnings
import sys
from typing import List, Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.utils.class_weight import compute_class_weight
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
import logging
from pathlib import Path
import json
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class EmailDataset(Dataset):
    """Custom Dataset class for email data"""
    
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

class PhishingDetector:
    """Main class for phishing detection system"""
    
    def __init__(self, model_name: str = "distilbert-base-uncased", max_length: int = 512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = None
        self.trainer = None
        
        # Check CUDA availability and quit if not available
        if not torch.cuda.is_available():
            logger.error("CUDA is not available! This system requires GPU acceleration.")
            logger.error("Please ensure you have:")
            logger.error("1. CUDA-compatible GPU (RTX 5060 Ti recommended)")
            logger.error("2. CUDA toolkit installed")
            logger.error("3. PyTorch with CUDA support installed")
            logger.error("Run: pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
            sys.exit(1)
        
        self.device = torch.device('cuda')
        logger.info(f"Using device: {self.device}")
        
        # RTX 5060 Ti optimization
        logger.info(f"CUDA available: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Optimize CUDA settings for RTX 5060 Ti
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Fix CUDA compatibility issues for RTX 5060 Ti
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Set CUDA device properties for RTX 5060 Ti
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            # Enable memory optimization
            torch.cuda.empty_cache()
            
            # Check compute capability
            props = torch.cuda.get_device_properties(0)
            logger.info(f"GPU Compute Capability: {props.major}.{props.minor}")
            
            # Set environment variables for CUDA compatibility
            import os
            os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
            os.environ['TORCH_USE_CUDA_DSA'] = '1'
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize email text"""
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', text)
        
        # Remove URLs (but keep the fact that URLs were present)
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls_found = len(re.findall(url_pattern, text))
        text = re.sub(url_pattern, '[URL]', text)
        
        # Remove email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        text = re.sub(email_pattern, '[EMAIL]', text)
        
        # Remove phone numbers
        phone_pattern = r'(\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}'
        text = re.sub(phone_pattern, '[PHONE]', text)
        
        # Remove excessive whitespace and normalize
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        # Add URL count as context if URLs were found
        if urls_found > 0:
            text = f"[{urls_found} URLs] {text}"
        
        return text
    
    def load_and_preprocess_data(self) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load and preprocess all datasets"""
        logger.info("Loading and preprocessing datasets...")
        
        datasets_info = {
            'CEAS_08.csv': {'label_col': 'label', 'text_cols': ['subject', 'body'], 'phishing_label': 1},
            'Enron.csv': {'label_col': 'label', 'text_cols': ['subject', 'body'], 'phishing_label': 0},
            'Ling.csv': {'label_col': 'label', 'text_cols': ['subject', 'body'], 'phishing_label': 0},
            'Nazario.csv': {'label_col': 'label', 'text_cols': ['subject', 'body'], 'phishing_label': 1},
            'Nigerian_Fraud.csv': {'label_col': 'label', 'text_cols': ['subject', 'body'], 'phishing_label': 1},
            'SpamAssasin.csv': {'label_col': 'label', 'text_cols': ['subject', 'body'], 'phishing_label': 0}
        }
        
        all_data = []
        dataset_stats = {}
        
        for filename, info in datasets_info.items():
            logger.info(f"Processing {filename}...")
            
            try:
                # Load dataset
                df = pd.read_csv(filename)
                
                # Store original stats
                dataset_stats[filename] = {
                    'original_rows': len(df),
                    'original_columns': list(df.columns),
                    'missing_data': df.isnull().sum().to_dict(),
                    'label_distribution': df[info['label_col']].value_counts().to_dict()
                }
                
                # Handle missing values
                df = df.dropna(subset=info['text_cols'] + [info['label_col']])
                
                # Combine text columns
                df['combined_text'] = df[info['text_cols']].fillna('').apply(
                    lambda x: ' '.join(x.astype(str)), axis=1
                )
                
                # Clean text
                df['cleaned_text'] = df['combined_text'].apply(self.clean_text)
                
                # Standardize labels (1 = phishing, 0 = legitimate)
                if info['phishing_label'] == 0:
                    df['binary_label'] = 1 - df[info['label_col']]  # Flip labels
                else:
                    df['binary_label'] = df[info['label_col']]
                
                # Filter out empty texts after cleaning
                df = df[df['cleaned_text'].str.len() > 10]
                
                # Select relevant columns
                processed_df = df[['cleaned_text', 'binary_label']].copy()
                processed_df['dataset'] = filename
                
                all_data.append(processed_df)
                
                logger.info(f"Processed {len(processed_df)} samples from {filename}")
                
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                continue
        
        # Combine all datasets
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Remove duplicates
        initial_size = len(combined_df)
        combined_df = combined_df.drop_duplicates(subset=['cleaned_text'])
        logger.info(f"Removed {initial_size - len(combined_df)} duplicate samples")
        
        # Final statistics
        dataset_stats['combined'] = {
            'total_samples': len(combined_df),
            'phishing_samples': combined_df['binary_label'].sum(),
            'legitimate_samples': len(combined_df) - combined_df['binary_label'].sum(),
            'class_balance': combined_df['binary_label'].mean()
        }
        
        logger.info(f"Total samples after preprocessing: {len(combined_df)}")
        logger.info(f"Phishing samples: {dataset_stats['combined']['phishing_samples']}")
        logger.info(f"Legitimate samples: {dataset_stats['combined']['legitimate_samples']}")
        logger.info(f"Class balance: {dataset_stats['combined']['class_balance']:.3f}")
        
        return combined_df, dataset_stats
    
    def create_data_loaders(self, df: pd.DataFrame, test_size: float = 0.2, val_size: float = 0.1):
        """Create train/validation/test data loaders"""
        logger.info("Creating data loaders...")
        
        # Split data
        train_df, temp_df = train_test_split(
            df, test_size=test_size + val_size, 
            random_state=42, stratify=df['binary_label']
        )
        
        val_df, test_df = train_test_split(
            temp_df, test_size=test_size/(test_size + val_size),
            random_state=42, stratify=temp_df['binary_label']
        )
        
        logger.info(f"Train samples: {len(train_df)}")
        logger.info(f"Validation samples: {len(val_df)}")
        logger.info(f"Test samples: {len(test_df)}")
        
        # Create datasets
        train_dataset = EmailDataset(
            train_df['cleaned_text'].tolist(),
            train_df['binary_label'].tolist(),
            self.tokenizer,
            self.max_length
        )
        
        val_dataset = EmailDataset(
            val_df['cleaned_text'].tolist(),
            val_df['binary_label'].tolist(),
            self.tokenizer,
            self.max_length
        )
        
        test_dataset = EmailDataset(
            test_df['cleaned_text'].tolist(),
            test_df['binary_label'].tolist(),
            self.tokenizer,
            self.max_length
        )
        
        return train_dataset, val_dataset, test_dataset, test_df
    
    def train_model(self, train_dataset, val_dataset, test_dataset, test_df):
        """Train the BERT model"""
        logger.info("Initializing model...")
        
        # Initialize model
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            problem_type="single_label_classification"
        )
        
        # Move model to device
        self.model.to(self.device)
        
        # Compute class weights for handling imbalance
        train_labels = [item['labels'].item() for item in train_dataset]
        class_weights = compute_class_weight(
            'balanced', 
            classes=np.unique(train_labels), 
            y=train_labels
        )
        class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.device)
        
        # Training arguments optimized for RTX 5060 Ti with Accelerate
        training_args = TrainingArguments(
            output_dir='./phishing_model',
            num_train_epochs=3,
            per_device_train_batch_size=16,  # Conservative for RTX 5060 Ti compatibility
            per_device_eval_batch_size=32,  # Conservative for RTX 5060 Ti compatibility
            warmup_steps=500,
            weight_decay=0.01,
            learning_rate=2e-5,
            logging_dir='./logs',
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            report_to=None,  # Disable wandb
            seed=42,
            fp16=False,  # Disabled for RTX 5060 Ti compatibility
            dataloader_pin_memory=True,
            dataloader_num_workers=4,  # Reduced to avoid Accelerate warnings
            gradient_accumulation_steps=1,
            gradient_checkpointing=True,  # Save memory
            dataloader_drop_last=False,  # Changed to False to avoid Accelerate warnings
            remove_unused_columns=False,
            push_to_hub=False,
            # Accelerate-specific optimizations
            prediction_loss_only=False,
        )
        
        # Define compute_metrics function
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=1)
            
            accuracy = accuracy_score(labels, predictions)
            precision = precision_score(labels, predictions, average='weighted', zero_division=0)
            recall = recall_score(labels, predictions, average='weighted', zero_division=0)
            f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
            
            return {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'eval_f1': f1  # Add this for the metric_for_best_model
            }
        
        # Custom trainer with class weights
        class WeightedTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                labels = inputs.get("labels")
                outputs = model(**inputs)
                logits = outputs.get("logits")
                
                loss_fct = nn.CrossEntropyLoss(weight=class_weights)
                loss = loss_fct(logits.view(-1, 2), labels.view(-1))
                
                return (loss, outputs) if return_outputs else loss
        
        # Initialize trainer
        self.trainer = WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train model with GPU optimization
        logger.info("Starting training...")
        
        # Clear GPU cache before training
        torch.cuda.empty_cache()
        
        # Monitor GPU memory usage
        if torch.cuda.is_available():
            logger.info(f"GPU memory before training: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
            logger.info(f"GPU memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")
        
        self.trainer.train()
        
        # Clear GPU cache after training
        torch.cuda.empty_cache()
        
        if torch.cuda.is_available():
            logger.info(f"GPU memory after training: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        
        # Evaluate on test set
        logger.info("Evaluating on test set...")
        test_results = self.trainer.evaluate(test_dataset)
        
        # Get predictions
        predictions = self.trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        y_true = test_df['binary_label'].values
        
        # Calculate metrics
        metrics = self.calculate_metrics(y_true, y_pred, predictions.predictions)
        
        return metrics, test_results
    
    def calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate comprehensive evaluation metrics"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'roc_auc': roc_auc_score(y_true, y_pred_proba[:, 1])
        }
        
        return metrics
    
    def create_visualizations(self, metrics, dataset_stats, test_df):
        """Create comprehensive visualizations"""
        logger.info("Creating visualizations...")
        
        # Set style
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Dataset distribution
        dataset_counts = {}
        for dataset in test_df['dataset'].unique():
            dataset_counts[dataset] = len(test_df[test_df['dataset'] == dataset])
        
        axes[0, 0].pie(dataset_counts.values(), labels=dataset_counts.keys(), autopct='%1.1f%%')
        axes[0, 0].set_title('Test Set Distribution by Dataset')
        
        # 2. Class distribution
        class_counts = test_df['binary_label'].value_counts()
        axes[0, 1].bar(['Legitimate', 'Phishing'], class_counts.values)
        axes[0, 1].set_title('Class Distribution in Test Set')
        axes[0, 1].set_ylabel('Count')
        
        # 3. Metrics comparison
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        bars = axes[1, 0].bar(metric_names, metric_values)
        axes[1, 0].set_title('Model Performance Metrics')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                          f'{value:.3f}', ha='center', va='bottom')
        
        # 4. Confusion Matrix
        y_pred = self.trainer.predict(EmailDataset(
            test_df['cleaned_text'].tolist(),
            test_df['binary_label'].tolist(),
            self.tokenizer,
            self.max_length
        )).predictions
        y_pred = np.argmax(y_pred, axis=1)
        
        cm = confusion_matrix(test_df['binary_label'], y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1],
                   xticklabels=['Legitimate', 'Phishing'],
                   yticklabels=['Legitimate', 'Phishing'])
        axes[1, 1].set_title('Confusion Matrix')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig('phishing_detection_results.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return fig
    
    def save_model(self, output_dir: str = './phishing_model'):
        """Save the trained model"""
        logger.info(f"Saving model to {output_dir}")
        self.trainer.save_model(output_dir)
        self.tokenizer.save_pretrained(output_dir)
    
    def generate_report(self, metrics, dataset_stats, test_results):
        """Generate comprehensive report"""
        report = {
            'timestamp': datetime.now().isoformat(),
            'model_info': {
                'model_name': self.model_name,
                'max_length': self.max_length,
                'device': str(self.device)
            },
            'dataset_statistics': dataset_stats,
            'performance_metrics': metrics,
            'test_results': test_results,
            'summary': {
                'total_samples': dataset_stats['combined']['total_samples'],
                'phishing_samples': dataset_stats['combined']['phishing_samples'],
                'legitimate_samples': dataset_stats['combined']['legitimate_samples'],
                'class_balance': dataset_stats['combined']['class_balance'],
                'accuracy': metrics['accuracy'],
                'f1_score': metrics['f1_score'],
                'roc_auc': metrics['roc_auc']
            }
        }
        
        # Save report
        with open('phishing_detection_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        return report

def main():
    """Main execution function"""
    logger.info("Starting Phishing Detection System")
    
    # Initialize detector
    detector = PhishingDetector()
    
    # Load and preprocess data
    df, dataset_stats = detector.load_and_preprocess_data()
    
    # Create data loaders
    train_dataset, val_dataset, test_dataset, test_df = detector.create_data_loaders(df)
    
    # Train model
    metrics, test_results = detector.train_model(train_dataset, val_dataset, test_dataset, test_df)
    
    # Create visualizations
    detector.create_visualizations(metrics, dataset_stats, test_df)
    
    # Save model
    detector.save_model()
    
    # Generate report
    report = detector.generate_report(metrics, dataset_stats, test_results)
    
    # Print summary
    print("\n" + "="*60)
    print("PHISHING DETECTION SYSTEM - FINAL RESULTS")
    print("="*60)
    print(f"Total Samples Processed: {report['summary']['total_samples']:,}")
    print(f"Phishing Samples: {report['summary']['phishing_samples']:,}")
    print(f"Legitimate Samples: {report['summary']['legitimate_samples']:,}")
    print(f"Class Balance: {report['summary']['class_balance']:.3f}")
    print("\nModel Performance:")
    print(f"Accuracy: {report['summary']['accuracy']:.4f}")
    print(f"F1-Score: {report['summary']['f1_score']:.4f}")
    print(f"ROC-AUC: {report['summary']['roc_auc']:.4f}")
    print("="*60)
    
    logger.info("Phishing Detection System completed successfully!")

if __name__ == "__main__":
    main()
