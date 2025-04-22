import os
import numpy as np
import pandas as pd
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Paths
BASE_DIR = r'C:\Users\kms99\Desktop\dephides_reproduction'
model_path = os.path.join(BASE_DIR, 'Reproduction & Analysis_model.h5')
tokenizer_path = os.path.join(BASE_DIR, 'Reproduction & Analysis_tokenizer.pkl')
input_path = os.path.join(BASE_DIR, 'test.txt')  # Your uploaded file
output_path = os.path.join(BASE_DIR, 'predicted_output.csv')

# Load model and tokenizer
model = load_model(model_path)
with open(tokenizer_path, 'rb') as f:
    tokenizer = pickle.load(f)

# Read data
true_labels, urls = [], []
with open(input_path, 'r', encoding='utf-8') as f:
    for line in f:
        parts = line.strip().split('\t')
        if len(parts) != 2:
            continue
        label, url = parts
        true_labels.append(label)
        urls.append(url)

# Preprocess URLs
sequences = tokenizer.texts_to_sequences(urls)
padded = pad_sequences(sequences, maxlen=200, padding='post', truncating='post')

# Predict
probs = model.predict(padded, batch_size=256)
predicted_labels = ['phishing' if np.argmax(p) == 1 else 'legitimate' for p in probs]
confidences = [round(float(np.max(p)), 4) for p in probs]

# Save to CSV
df = pd.DataFrame({
    'True Label': true_labels, 
    'URL': urls,
    'Predicted Label': predicted_labels,
    'Confidence': confidences
})
df.to_csv(output_path, index=False)
print(f"✅ Predictions saved to: {output_path}")
