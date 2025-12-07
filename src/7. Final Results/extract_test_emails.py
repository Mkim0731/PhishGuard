#!/usr/bin/env python3
"""
Extract Sample Phishing Emails for Testing
==========================================
Extracts sample emails from training datasets for testing the Chrome extension.
"""

import pandas as pd
import random
from pathlib import Path

def extract_samples(csv_file, num_samples=5, is_phishing=True):
    """Extract sample emails from a CSV file"""
    try:
        df = pd.read_csv(csv_file)
        print(f"\nReading {csv_file.name}...")
        print(f"  Total rows: {len(df)}")
        
        # Filter for phishing emails if needed
        if 'label' in df.columns:
            if is_phishing:
                df = df[df['label'] == 1]
            else:
                df = df[df['label'] == 0]
        
        if len(df) == 0:
            print(f"  No {'phishing' if is_phishing else 'legitimate'} emails found")
            return []
        
        # Get random samples
        num_samples = min(num_samples, len(df))
        samples = df.sample(n=num_samples, random_state=42)
        
        results = []
        for idx, row in samples.iterrows():
            subject = str(row.get('subject', ''))
            body = str(row.get('body', ''))
            
            # Combine subject and body
            email_text = f"Subject: {subject}\n\n{body}"
            results.append({
                'source': csv_file.name,
                'email': email_text,
                'is_phishing': is_phishing
            })
        
        print(f"  Extracted {len(results)} samples")
        return results
        
    except Exception as e:
        print(f"  Error reading {csv_file.name}: {e}")
        return []

def main():
    script_dir = Path(__file__).parent
    model1_dir = script_dir / 'Model1_EmailContent'
    
    print("="*60)
    print("Extracting Sample Emails for Testing")
    print("="*60)
    
    # Phishing datasets
    phishing_files = [
        model1_dir / 'CEAS_08.csv',
        model1_dir / 'Nazario.csv',
        model1_dir / 'Nigerian_Fraud.csv',
        model1_dir / 'phishing_email.csv'
    ]
    
    # Legitimate datasets
    legitimate_files = [
        model1_dir / 'Enron.csv',
        model1_dir / 'Ling.csv',
        model1_dir / 'SpamAssasin.csv'
    ]
    
    all_samples = []
    
    # Extract phishing samples
    print("\n--- Phishing Email Samples ---")
    for csv_file in phishing_files:
        if csv_file.exists():
            samples = extract_samples(csv_file, num_samples=3, is_phishing=True)
            all_samples.extend(samples)
        else:
            print(f"\n{csv_file.name} not found, skipping...")
    
    # Extract legitimate samples
    print("\n--- Legitimate Email Samples ---")
    for csv_file in legitimate_files:
        if csv_file.exists():
            samples = extract_samples(csv_file, num_samples=2, is_phishing=False)
            all_samples.extend(samples)
        else:
            print(f"\n{csv_file.name} not found, skipping...")
    
    # Save to text files
    output_dir = script_dir / 'test_emails'
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*60)
    print("Saving samples to files...")
    print("="*60)
    
    for i, sample in enumerate(all_samples, 1):
        label = "phishing" if sample['is_phishing'] else "legitimate"
        filename = output_dir / f"sample_{i:02d}_{label}_{sample['source'].replace('.csv', '')}.txt"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"Source: {sample['source']}\n")
            f.write(f"Type: {label.upper()}\n")
            f.write("="*60 + "\n\n")
            f.write(sample['email'])
        
        print(f"  Saved: {filename.name}")
    
    print(f"\nOK: Extracted {len(all_samples)} sample emails")
    print(f"Location: {output_dir}")
    print("\nYou can now copy these emails and paste them into the Chrome extension!")

if __name__ == "__main__":
    main()

