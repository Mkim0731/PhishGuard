# BERT Model Training for Phishing Detection
# Advanced machine learning pipeline with DistilBERT

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

class EmailDataset(Dataset):
    """
    Custom PyTorch Dataset for email classification
    """
    
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        
        # Tokenize text
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

class PhishingBERTTrainer:
    """
    BERT-based trainer for phishing email detection
    """
    
    def __init__(self, model_name='distilbert-base-uncased', max_length=512):
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = None
        self.model = None
        self.trainer = None
        
    def setup_model(self, num_labels=2):
        """
        Initialize tokenizer and model
        """
        print(f"Setting up {self.model_name} model...")
        
        # Check for GPU availability
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Load tokenizer
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_name)
        
        # Load model
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            problem_type="single_label_classification"
        )
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to(device)
            print("Model moved to GPU")
        
        print(f"SUCCESS: Model loaded: {self.model.num_parameters():,} parameters")
        
    def create_datasets(self, train_texts, train_labels, val_texts, val_labels, test_texts, test_labels):
        """
        Create PyTorch datasets for training
        """
        print("Creating PyTorch datasets...")
        
        train_dataset = EmailDataset(train_texts, train_labels, self.tokenizer, self.max_length)
        val_dataset = EmailDataset(val_texts, val_labels, self.tokenizer, self.max_length)
        test_dataset = EmailDataset(test_texts, test_labels, self.tokenizer, self.max_length)
        
        print(f"SUCCESS: Datasets created:")
        print(f"  Train: {len(train_dataset)} samples")
        print(f"  Validation: {len(val_dataset)} samples")
        print(f"  Test: {len(test_dataset)} samples")
        
        return train_dataset, val_dataset, test_dataset
    
    def compute_metrics(self, eval_pred):
        """
        Compute evaluation metrics
        """
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        accuracy = accuracy_score(labels, predictions)
        
        # Calculate ROC-AUC
        try:
            # Get prediction probabilities for ROC-AUC
            predictions_proba = torch.softmax(torch.tensor(eval_pred[0]), dim=1)[:, 1].numpy()
            roc_auc = roc_auc_score(labels, predictions_proba)
        except:
            roc_auc = 0.0
        
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'roc_auc': roc_auc
        }
    
    def train_model(self, train_dataset, val_dataset, output_dir='./phishing_model'):
        """
        Train the BERT model
        """
        print("Starting model training...")
        
        # Calculate class weights for imbalanced data
        train_labels = [train_dataset[i]['labels'].item() for i in range(len(train_dataset))]
        class_counts = np.bincount(train_labels)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * len(class_counts)
        
        print(f"Class weights: {class_weights}")
        
        # Check for GPU availability
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Training arguments optimized for GPU
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=32 if torch.cuda.is_available() else 16,  # Larger batch size for GPU
            per_device_eval_batch_size=64 if torch.cuda.is_available() else 32,   # Larger eval batch for GPU
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir=f'{output_dir}/logs',
            logging_steps=100,
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to=None,  # Disable wandb/tensorboard
            seed=42,
            dataloader_num_workers=0,  # Disable multiprocessing to avoid Windows issues
            fp16=True if torch.cuda.is_available() else False,  # Enable mixed precision for GPU
            dataloader_pin_memory=True if torch.cuda.is_available() else False,  # Pin memory for GPU
            remove_unused_columns=False,
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train the model
        print("Training started...")
        training_result = self.trainer.train()
        
        print("SUCCESS: Training completed!")
        print(f"Final training loss: {training_result.training_loss:.4f}")
        
        return training_result
    
    def evaluate_model(self, test_dataset):
        """
        Evaluate the trained model
        """
        print("Evaluating model on test set...")
        
        # Get predictions
        predictions = self.trainer.predict(test_dataset)
        pred_labels = np.argmax(predictions.predictions, axis=1)
        true_labels = predictions.label_ids
        
        # Calculate metrics
        metrics = self.compute_metrics((predictions.predictions, true_labels))
        
        print("Test Set Results:")
        print(f"  Accuracy: {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall: {metrics['recall']:.4f}")
        print(f"  F1-Score: {metrics['f1']:.4f}")
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
        
        return metrics, pred_labels, true_labels
    
    def save_model(self, save_path='./phishing_bert_model'):
        """
        Save the trained model and tokenizer
        """
        print(f"Saving model to {save_path}...")
        
        self.model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)
        
        print("SUCCESS: Model saved successfully!")
    
    def load_model(self, model_path='./phishing_bert_model'):
        """
        Load a pre-trained model
        """
        print(f"Loading model from {model_path}...")
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_path)
        
        print("SUCCESS: Model loaded successfully!")

def plot_training_metrics(trainer):
    """
    Plot training metrics
    """
    print("Plotting training metrics...")
    
    # Get training history
    log_history = trainer.state.log_history
    
    # Extract metrics
    train_losses = [log['loss'] for log in log_history if 'loss' in log]
    eval_losses = [log['eval_loss'] for log in log_history if 'eval_loss' in log]
    eval_f1 = [log['eval_f1'] for log in log_history if 'eval_f1' in log]
    eval_accuracy = [log['eval_accuracy'] for log in log_history if 'eval_accuracy' in log]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss plot
    axes[0, 0].plot(train_losses, label='Training Loss', color='blue')
    axes[0, 0].plot(eval_losses, label='Validation Loss', color='red')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].set_xlabel('Steps')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # F1 Score plot
    axes[0, 1].plot(eval_f1, label='Validation F1', color='green')
    axes[0, 1].set_title('Validation F1 Score')
    axes[0, 1].set_xlabel('Steps')
    axes[0, 1].set_ylabel('F1 Score')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Accuracy plot
    axes[1, 0].plot(eval_accuracy, label='Validation Accuracy', color='purple')
    axes[1, 0].set_title('Validation Accuracy')
    axes[1, 0].set_xlabel('Steps')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Combined metrics
    axes[1, 1].plot(eval_f1, label='F1 Score', color='green')
    axes[1, 1].plot(eval_accuracy, label='Accuracy', color='purple')
    axes[1, 1].set_title('Validation Metrics Comparison')
    axes[1, 1].set_xlabel('Steps')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('training_metrics.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names=['Legitimate', 'Phishing']):
    """
    Plot confusion matrix
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == "__main__":
    # Load the preprocessed data
    import sys
    import importlib.util
    import os

    # Check if processed data exists, if not run preprocessing
    if os.path.exists('processed_train_val_test_splits.csv'):
        print("Found processed data files. Loading from CSV...")
        # Import the preprocessing module
        spec = importlib.util.spec_from_file_location("data_preprocessing", "01_data_preprocessing.py")
        data_preprocessing = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(data_preprocessing)
        
        processor = data_preprocessing.EmailDataProcessor()
        splits = processor.load_processed_data()
        
        if splits is None:
            print("Failed to load processed data. Running preprocessing...")
            datasets = processor.load_datasets()
            processor.analyze_datasets()
            processed_data = processor.preprocess_data()
            splits = processor.create_train_test_split()
    else:
        print("No processed data found. Running preprocessing...")
        # Import the preprocessing module
        spec = importlib.util.spec_from_file_location("data_preprocessing", "01_data_preprocessing.py")
        data_preprocessing = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(data_preprocessing)
        
        processor = data_preprocessing.EmailDataProcessor()
        datasets = processor.load_datasets()
        processor.analyze_datasets()
        processed_data = processor.preprocess_data()
        splits = processor.create_train_test_split()

    print("Initializing BERT Trainer...")
    trainer = PhishingBERTTrainer()

    # Setup model
    trainer.setup_model()

    # Create datasets
    train_dataset, val_dataset, test_dataset = trainer.create_datasets(
        splits['train']['text'], splits['train']['label'],
        splits['val']['text'], splits['val']['label'],
        splits['test']['text'], splits['test']['label']
    )

    # Train the model
    training_result = trainer.train_model(train_dataset, val_dataset)

    # Evaluate the model
    test_metrics, pred_labels, true_labels = trainer.evaluate_model(test_dataset)

    # Plot results
    plot_training_metrics(trainer.trainer)
    plot_confusion_matrix(true_labels, pred_labels)

    # Save the model
    trainer.save_model()

    print("\nBERT Model Training Completed Successfully!")
    print("=" * 60)
