# Model Evaluation and Analysis
# Comprehensive performance analysis and visualization

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve
)
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import warnings
warnings.filterwarnings('ignore')

class ModelEvaluator:
    """
    Comprehensive model evaluation and analysis
    """
    
    def __init__(self, model_path='./phishing_bert_model'):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        
    def load_model(self):
        """
        Load the trained model
        """
        print(f"Loading model from {self.model_path}...")
        
        # Check for GPU availability
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(self.model_path)
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            self.model = self.model.to(device)
            print("Model moved to GPU")
        
        print("SUCCESS: Model loaded successfully!")
    
    def predict_single_email(self, email_text):
        """
        Predict phishing probability for a single email
        """
        if self.model is None:
            self.load_model()
        
        # Tokenize input
        inputs = self.tokenizer(
            email_text,
            truncation=True,
            padding='max_length',
            max_length=512,
            return_tensors='pt'
        )
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][prediction].item()
        
        return {
            'prediction': prediction,
            'confidence': confidence,
            'phishing_probability': probabilities[0][1].item(),
            'legitimate_probability': probabilities[0][0].item()
        }
    
    def evaluate_on_test_set(self, test_texts, test_labels):
        """
        Comprehensive evaluation on test set
        """
        print("Evaluating model on test set...")
        
        predictions = []
        probabilities = []
        
        for text in test_texts:
            result = self.predict_single_email(text)
            predictions.append(result['prediction'])
            probabilities.append(result['phishing_probability'])
        
        predictions = np.array(predictions)
        probabilities = np.array(probabilities)
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predictions)
        precision = precision_score(test_labels, predictions)
        recall = recall_score(test_labels, predictions)
        f1 = f1_score(test_labels, predictions)
        roc_auc = roc_auc_score(test_labels, probabilities)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        
        print("Test Set Performance:")
        for metric, value in metrics.items():
            print(f"  {metric.capitalize()}: {value:.4f}")
        
        return metrics, predictions, probabilities
    
    def plot_performance_metrics(self, metrics):
        """
        Create comprehensive performance visualizations
        """
        print("Creating performance visualizations...")
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Performance Metrics', 'ROC Curve', 'Precision-Recall Curve', 'Confusion Matrix'),
            specs=[[{"type": "bar"}, {"type": "scatter"}],
                   [{"type": "scatter"}, {"type": "heatmap"}]]
        )
        
        # Performance metrics bar chart
        metric_names = list(metrics.keys())
        metric_values = list(metrics.values())
        
        fig.add_trace(
            go.Bar(x=metric_names, y=metric_values, name='Metrics'),
            row=1, col=1
        )
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(test_labels, probabilities)
        fig.add_trace(
            go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC Curve'),
            row=1, col=2
        )
        
        # Precision-Recall Curve
        precision_vals, recall_vals, _ = precision_recall_curve(test_labels, probabilities)
        fig.add_trace(
            go.Scatter(x=recall_vals, y=precision_vals, mode='lines', name='PR Curve'),
            row=2, col=1
        )
        
        # Confusion Matrix
        cm = confusion_matrix(test_labels, predictions)
        fig.add_trace(
            go.Heatmap(z=cm, x=['Legitimate', 'Phishing'], y=['Legitimate', 'Phishing']),
            row=2, col=2
        )
        
        fig.update_layout(height=800, showlegend=False, title_text="Model Performance Analysis")
        fig.write_html("model_performance.html")
        fig.show()
    
    def analyze_prediction_errors(self, test_texts, test_labels, predictions):
        """
        Analyze cases where the model made incorrect predictions
        """
        print("Analyzing prediction errors...")
        
        errors = []
        for i, (text, true_label, pred_label) in enumerate(zip(test_texts, test_labels, predictions)):
            if true_label != pred_label:
                errors.append({
                    'index': i,
                    'text': text[:200] + '...' if len(text) > 200 else text,
                    'true_label': 'Phishing' if true_label == 1 else 'Legitimate',
                    'predicted_label': 'Phishing' if pred_label == 1 else 'Legitimate',
                    'text_length': len(text)
                })
        
        error_df = pd.DataFrame(errors)
        
        print(f"Error Analysis:")
        print(f"  Total errors: {len(errors)}")
        print(f"  False Positives: {len(error_df[error_df['true_label'] == 'Legitimate'])}")
        print(f"  False Negatives: {len(error_df[error_df['true_label'] == 'Phishing'])}")
        
        return error_df
    
    def generate_classification_report(self, test_labels, predictions):
        """
        Generate detailed classification report
        """
        print("Generating classification report...")
        
        report = classification_report(
            test_labels, predictions, 
            target_names=['Legitimate', 'Phishing'],
            output_dict=True
        )
        
        print("\nDetailed Classification Report:")
        print(classification_report(test_labels, predictions, target_names=['Legitimate', 'Phishing']))
        
        return report

def create_performance_summary(metrics, error_analysis):
    """
    Create a comprehensive performance summary
    """
    print("\n" + "="*60)
    print("FINAL MODEL PERFORMANCE SUMMARY")
    print("="*60)
    
    print(f"\nCore Metrics:")
    print(f"  Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision: {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall:    {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  F1-Score:  {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f} ({metrics['roc_auc']*100:.2f}%)")
    
    print(f"\nError Analysis:")
    print(f"  Total Errors: {len(error_analysis)}")
    print(f"  False Positives: {len(error_analysis[error_analysis['true_label'] == 'Legitimate'])}")
    print(f"  False Negatives: {len(error_analysis[error_analysis['true_label'] == 'Phishing'])}")
    
    print(f"\nModel Performance Assessment:")
    if metrics['f1_score'] > 0.9:
        print("  EXCELLENT: Model shows outstanding performance!")
    elif metrics['f1_score'] > 0.8:
        print("  VERY GOOD: Model shows strong performance!")
    elif metrics['f1_score'] > 0.7:
        print("  GOOD: Model shows solid performance!")
    else:
        print("  NEEDS IMPROVEMENT: Model performance could be enhanced!")
    
    print("\n" + "="*60)

# Load test data
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
    
    processor = data_preprocessing.processor
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
    
    processor = data_preprocessing.processor
    datasets = processor.load_datasets()
    processor.analyze_datasets()
    processed_data = processor.preprocess_data()
    splits = processor.create_train_test_split()

# Initialize evaluator
evaluator = ModelEvaluator()

# Evaluate on test set
test_texts = splits['test']['text']
test_labels = splits['test']['label']

metrics, predictions, probabilities = evaluator.evaluate_on_test_set(test_texts, test_labels)

# Generate visualizations
evaluator.plot_performance_metrics(metrics)

# Analyze errors
error_analysis = evaluator.analyze_prediction_errors(test_texts, test_labels, predictions)

# Generate classification report
classification_report = evaluator.generate_classification_report(test_labels, predictions)

# Create performance summary
create_performance_summary(metrics, error_analysis)

print("\nModel Evaluation Completed Successfully!")
