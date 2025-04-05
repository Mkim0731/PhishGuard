# AI-Powered Email Screening

## Goal
Automatically detect phishing emails by analyzing email content, sender metadata, and structure using machine learning.

## Existing Technologies & Methodologies


### 1. Gmail’s Spam & Phishing Detection System 
- NLP, metadata analysis, and template recognition, (Related library that can be possibly used for the project-SpaCy&NLTK)
** Strengths:**
- Uses billions of emails to train multi layered models(TF-IDF, deep learning, transformers) 
- Text Preprocessing - Tokenization(break text into smaller units), lemmatization(clean irrelevant words, standardize format) 
- Feature extraction - Entity recognitions(extract named entities)
- Contextual Triggers & Templates - detects template-based phishing
** Weaknesses:**
- Operates as a black box—hard to interpret or debug model decisions externally. Requires enormous data

---

### 2. BERT / RoBERTa via HuggingFace
- Transformer-based deep learning models pretrained on large corpora.
- Can be fine-tuned for phishing email classification.

**Strengths:**
- Captures deep context and word relationships.
- High accuracy in text classification tasks.

**Weaknesses:**
- Requires significant computational resources (GPU).
- Slower inference for real-time apps.

**GitHub:** [HuggingFace Transformers](https://github.com/huggingface/transformers)

---

### 3. NLP Preprocessing: SpaCy / NLTK
- Libraries used for text preprocessing (tokenization, lemmatization, etc.)

**Strengths:**
- Lightweight and fast.
- Easy to use with scikit-learn pipelines.

**Weaknesses:**
- Not classifiers; only preprocessing tools.

**GitHub:** [SpaCy](https://github.com/explosion/spaCy) | [NLTK](https://github.com/nltk/nltk)
