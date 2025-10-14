# Phishing Email Detection System
# Data Preprocessing and BERT Model Training

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import nltk
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    DistilBertTokenizer,
    DistilBertForSequenceClassification
)
from datasets import Dataset as HFDataset
import warnings
warnings.filterwarnings('ignore')

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)
except:
    pass

class EmailDataProcessor:
    """
    Comprehensive data processor for phishing email detection
    Handles multiple dataset formats and preprocessing steps
    """
    
    def __init__(self):
        self.datasets = {}
        self.combined_data = None
        self.processed_data = None
        
    def load_datasets(self):
        """
        Load all 6 datasets and standardize their format
        """
        print("Loading and analyzing datasets...")
        
        # Dataset configurations
        dataset_configs = {
            'Nigerian_Fraud.csv': {
                'text_cols': ['subject', 'body'],
                'label_col': 'label',
                'label_mapping': None  # Already binary
            },
            'Enron.csv': {
                'text_cols': ['subject', 'body'],
                'label_col': 'label',
                'label_mapping': None  # Already binary
            },
            'SpamAssasin.csv': {
                'text_cols': ['subject', 'body'],
                'label_col': 'label',
                'label_mapping': None  # Already binary
            },
            'CEAS_08.csv': {
                'text_cols': ['subject', 'body'],
                'label_col': 'label',
                'label_mapping': None  # Already binary
            },
            'Ling.csv': {
                'text_cols': ['subject', 'body'],
                'label_col': 'label',
                'label_mapping': None  # Already binary
            },
            'Nazario.csv': {
                'text_cols': ['subject', 'body'],
                'label_col': 'label',
                'label_mapping': None  # Already binary
            }
        }
        
        for filename, config in dataset_configs.items():
            try:
                print(f"Loading {filename}...")
                df = pd.read_csv(filename)
                
                # Combine text columns
                text_cols = config['text_cols']
                df['combined_text'] = df[text_cols].fillna('').astype(str).agg(' '.join, axis=1)
                
                # Standardize label column
                df['label'] = df[config['label_col']].astype(int)
                
                # Keep only necessary columns
                df = df[['combined_text', 'label']].copy()
                
                # Remove rows with empty text
                df = df[df['combined_text'].str.strip() != ''].copy()
                
                self.datasets[filename] = df
                print(f"SUCCESS: {filename}: {len(df)} samples, {df['label'].value_counts().to_dict()}")
                
            except Exception as e:
                print(f"ERROR: Error loading {filename}: {str(e)}")
        
        return self.datasets
    
    def analyze_datasets(self):
        """
        Comprehensive dataset analysis
        """
        print("\nDataset Analysis Summary:")
        print("=" * 50)
        
        total_samples = 0
        total_phishing = 0
        total_legitimate = 0
        
        for name, df in self.datasets.items():
            samples = len(df)
            phishing = df['label'].sum()
            legitimate = samples - phishing
            
            total_samples += samples
            total_phishing += phishing
            total_legitimate += legitimate
            
            print(f"\n{name}:")
            print(f"  Total samples: {samples:,}")
            print(f"  Phishing: {phishing:,} ({phishing/samples*100:.1f}%)")
            print(f"  Legitimate: {legitimate:,} ({legitimate/samples*100:.1f}%)")
            print(f"  Avg text length: {df['combined_text'].str.len().mean():.0f} chars")
        
        print(f"\nCOMBINED DATASET:")
        print(f"  Total samples: {total_samples:,}")
        print(f"  Phishing: {total_phishing:,} ({total_phishing/total_samples*100:.1f}%)")
        print(f"  Legitimate: {total_legitimate:,} ({total_legitimate/total_samples*100:.1f}%)")
        
        return {
            'total_samples': total_samples,
            'total_phishing': total_phishing,
            'total_legitimate': total_legitimate,
            'phishing_ratio': total_phishing / total_samples
        }
    
    def combine_datasets(self):
        """
        Combine all datasets into a single dataframe
        """
        print("\nCombining datasets...")
        
        all_dataframes = []
        for name, df in self.datasets.items():
            df_copy = df.copy()
            df_copy['source_dataset'] = name
            all_dataframes.append(df_copy)
        
        self.combined_data = pd.concat(all_dataframes, ignore_index=True)
        print(f"SUCCESS: Combined dataset: {len(self.combined_data):,} samples")
        
        return self.combined_data
    
    def clean_text(self, text):
        """
        Comprehensive text cleaning for email content
        """
        if pd.isna(text) or text == '':
            return ''
        
        # Convert to string
        text = str(text)
        
        # Remove HTML tags
        soup = BeautifulSoup(text, 'html.parser')
        text = soup.get_text()
        
        # Remove email headers and metadata
        text = re.sub(r'From:.*?\n', '', text, flags=re.IGNORECASE)
        text = re.sub(r'To:.*?\n', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Subject:.*?\n', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Date:.*?\n', '', text, flags=re.IGNORECASE)
        text = re.sub(r'Message-ID:.*?\n', '', text, flags=re.IGNORECASE)
        
        # Remove URLs (but keep the fact that URLs were present)
        url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        urls_found = len(re.findall(url_pattern, text))
        text = re.sub(url_pattern, '[URL]', text)
        
        # Remove email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        text = re.sub(email_pattern, '[EMAIL]', text)
        
        # Remove phone numbers
        phone_pattern = r'(\+?1[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        text = re.sub(phone_pattern, '[PHONE]', text)
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\(\)\[\]]', ' ', text)
        
        # Clean up whitespace again
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def preprocess_data(self):
        """
        Complete preprocessing pipeline
        """
        print("\nStarting data preprocessing...")
        
        if self.combined_data is None:
            self.combine_datasets()
        
        df = self.combined_data.copy()
        
        print("Cleaning text data...")
        df['cleaned_text'] = df['combined_text'].apply(self.clean_text)
        
        # Remove empty texts after cleaning
        df = df[df['cleaned_text'].str.strip() != ''].copy()
        
        # Remove duplicates
        initial_count = len(df)
        df = df.drop_duplicates(subset=['cleaned_text']).copy()
        duplicates_removed = initial_count - len(df)
        print(f"Removed {duplicates_removed:,} duplicate samples")
        
        # Balance classes (optional - for better training)
        phishing_samples = df[df['label'] == 1]
        legitimate_samples = df[df['label'] == 0]
        
        # If severe imbalance, we'll handle it during training with class weights
        print(f"Class distribution after cleaning:")
        print(f"  Phishing: {len(phishing_samples):,}")
        print(f"  Legitimate: {len(legitimate_samples):,}")
        
        self.processed_data = df
        print(f"SUCCESS: Preprocessing complete: {len(df):,} samples ready for training")
        
        return df
    
    def create_train_test_split(self, test_size=0.2, val_size=0.1):
        """
        Create stratified train/validation/test splits
        """
        print(f"\nCreating train/validation/test splits...")
        
        if self.processed_data is None:
            self.preprocess_data()
        
        df = self.processed_data.copy()
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            df['cleaned_text'], df['label'], 
            test_size=test_size, 
            random_state=42, 
            stratify=df['label']
        )
        
        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, 
            test_size=val_size/(1-test_size), 
            random_state=42, 
            stratify=y_temp
        )
        
        splits = {
            'train': {'text': X_train, 'label': y_train},
            'val': {'text': X_val, 'label': y_val},
            'test': {'text': X_test, 'label': y_test}
        }
        
        print(f"Split sizes:")
        for split_name, split_data in splits.items():
            print(f"  {split_name}: {len(split_data['text']):,} samples")
        
        # Save processed data to CSV files
        print("Saving processed datasets...")
        
        # Save individual processed datasets
        for name, df in self.datasets.items():
            processed_filename = f"processed_{name}"
            df.to_csv(processed_filename, index=False)
            print(f"Saved: {processed_filename}")
        
        # Save combined processed dataset
        self.combined_data.to_csv('processed_combined_dataset.csv', index=False)
        print("Saved: processed_combined_dataset.csv")
        
        # Save train/validation/test splits
        splits_df = pd.DataFrame({
            'text': pd.concat([splits['train']['text'], splits['val']['text'], splits['test']['text']], ignore_index=True),
            'label': pd.concat([splits['train']['label'], splits['val']['label'], splits['test']['label']], ignore_index=True),
            'split': (['train'] * len(splits['train']['text']) + 
                     ['val'] * len(splits['val']['text']) + 
                     ['test'] * len(splits['test']['text']))
        })
        splits_df.to_csv('processed_train_val_test_splits.csv', index=False)
        print("Saved: processed_train_val_test_splits.csv")
        
        return splits
    
    def load_processed_data(self):
        """
        Load previously processed data from CSV files
        """
        print("Loading processed datasets from CSV files...")
        
        try:
            # Load train/validation/test splits
            splits_df = pd.read_csv('processed_train_val_test_splits.csv')
            
            # Recreate splits dictionary
            splits = {}
            for split_name in ['train', 'val', 'test']:
                split_data = splits_df[splits_df['split'] == split_name]
                splits[split_name] = {
                    'text': split_data['text'],
                    'label': split_data['label']
                }
            
            print(f"SUCCESS: Loaded processed data:")
            print(f"  Train: {len(splits['train']['text']):,} samples")
            print(f"  Validation: {len(splits['val']['text']):,} samples")
            print(f"  Test: {len(splits['test']['text']):,} samples")
            
            return splits
            
        except FileNotFoundError:
            print("ERROR: Processed data files not found. Please run preprocessing first.")
            return None

# Initialize processor
processor = EmailDataProcessor()

# Load and analyze datasets
datasets = processor.load_datasets()
analysis = processor.analyze_datasets()

# Preprocess data
processed_data = processor.preprocess_data()

# Create train/test splits
splits = processor.create_train_test_split()

print("\nData preprocessing completed successfully!")
print("=" * 60)
