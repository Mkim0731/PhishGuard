import os
import random
import numpy as np
import tensorflow as tf
import requests
import csv
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Conv1D,
    MaxPooling1D, GlobalMaxPooling1D,
    Dense, Dropout
)
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -----------------------------
# 0. Auto-Download Dataset if Missing
# -----------------------------
def download_if_missing(local_path, url):
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    if not os.path.exists(local_path):
        print(f"Downloading {os.path.basename(local_path)}...")
        r = requests.get(url, stream=True)
        r.raise_for_status()
        with open(local_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

base_url = 'https://raw.githubusercontent.com/ebubekirbbr/dephides/main/dataset/small_dataset'
files = {
    'dataset/small_dataset/train.txt': f"{base_url}/train.txt",
    'dataset/small_dataset/val.txt':   f"{base_url}/val.txt",
    'dataset/small_dataset/test.txt':  f"{base_url}/test.txt",
}
for local, url in files.items():
    download_if_missing(local, url)

# -----------------------------
# 1. Reproducibility Settings
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# 2. Configuration Parameters
# -----------------------------
DATASET_DIR = 'dataset/small_dataset'
MAX_LEN     = 200
EMBED_DIM   = 50
FILTERS     = 128
KERNEL_SIZES = [3, 4, 5]
POOL_SIZES   = [2, 2, 2]
DROPOUT_RATE = 0.5
BATCH_SIZE   = 1000
EPOCHS       = 30

# -----------------------------
# 3. Data Loading & Preprocessing
# -----------------------------
def load_data(path):
    urls, labels = [], []
    with open(path, newline='') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) != 2:
                continue
            label_str, url = row
            label = 0 if label_str.strip().lower() == 'legitimate' else 1
            urls.append(url)
            labels.append(label)
    return np.array(urls), np.array(labels)

train_urls, train_labels = load_data(os.path.join(DATASET_DIR, 'train.txt'))
val_urls,   val_labels   = load_data(os.path.join(DATASET_DIR, 'val.txt'))
test_urls,  test_labels  = load_data(os.path.join(DATASET_DIR, 'test.txt'))

tokenizer = Tokenizer(char_level=True, oov_token='<OOV>')
tokenizer.fit_on_texts(train_urls)
vocab_size = len(tokenizer.word_index) + 1
print(f"Character vocab size: {vocab_size}")

def preprocess(texts):
    seqs = tokenizer.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=MAX_LEN, padding='post', truncating='post')

X_train = preprocess(train_urls)
X_val   = preprocess(val_urls)
X_test  = preprocess(test_urls)

# -----------------------------
# 4. Model Definition
# -----------------------------
inputs = Input(shape=(MAX_LEN,))
x = Embedding(input_dim=vocab_size, output_dim=EMBED_DIM, input_length=MAX_LEN)(inputs)
for i, ks in enumerate(KERNEL_SIZES):
    x = Conv1D(filters=FILTERS, kernel_size=ks, activation='relu', padding='same')(x)
    x = MaxPooling1D(pool_size=POOL_SIZES[i], padding='same')(x)
x = GlobalMaxPooling1D()(x)
x = Dropout(DROPOUT_RATE)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(DROPOUT_RATE)(x)
outputs = Dense(2, activation='softmax')(x)

model = Model(inputs=inputs, outputs=outputs)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.summary()

# -----------------------------
# 5. Training
# -----------------------------
history = model.fit(X_train, train_labels,
                    validation_data=(X_val, val_labels),
                    epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=1)

# -----------------------------
# 6. Evaluation
# -----------------------------
pred_probs = model.predict(X_test, batch_size=BATCH_SIZE)
pred_labels = np.argmax(pred_probs, axis=1)
acc = accuracy_score(test_labels, pred_labels)
print(f"Test accuracy: {acc:.4f}")
print("Classification Report:")
print(classification_report(test_labels, pred_labels, target_names=['benign', 'phishing']))

# -----------------------------
# 7. Save Artifacts
# -----------------------------
SAVE_DIR = 'dephides_reproduction'
os.makedirs(SAVE_DIR, exist_ok=True)
model.save(os.path.join(SAVE_DIR, 'model.h5'))
with open(os.path.join(SAVE_DIR, 'tokenizer.pkl'), 'wb') as f:
    pickle.dump(tokenizer, f)

# -----------------------------
# 8. Plot Training Curves
# -----------------------------
plt.figure(figsize=(10, 4))

# Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

# Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'training_curves.png'))
plt.show()

# -----------------------------
# 9. Plot Confusion Matrix
# -----------------------------
cm = confusion_matrix(test_labels, pred_labels)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Phishing'],
            yticklabels=['Benign', 'Phishing'])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrix.png'))
plt.show()

print(f"Artifacts saved to {SAVE_DIR}/")
