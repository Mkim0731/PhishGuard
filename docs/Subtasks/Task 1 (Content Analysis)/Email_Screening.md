# AI-Powered Email Screening

## Goal
Automatically detect phishing emails by analyzing email content, sender metadata, and structure using machine learning.

## Existing Technologies & Methodologies

### 1. scikit-learn
- Used for traditional ML models like Logistic Regression, Random Forest, and XGBoost.
- Works well with structured features (e.g., word frequency, header flags).

**Strengths:**
- Fast training and testing.
- Interpretable models.
- Easy integration with Python-based pipelines.

**Weaknesses:**
- Requires manual feature engineering.
- Doesn’t capture semantic context of words.

**GitHub:** [scikit-learn](https://github.com/scikit-learn/scikit-learn)

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
