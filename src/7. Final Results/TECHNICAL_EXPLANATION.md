# PhishGuard: Technical Explanation for Academic Presentation

## Overview

PhishGuard is a dual-model AI system that detects phishing emails and spam by analyzing both email content and embedded URLs. The system uses two complementary machine learning models that work together to provide comprehensive threat detection.

---

## System Architecture

### Two-Model Approach

**Why Two Models?**
- **Complementary Analysis**: Email content and URLs provide different signals
- **Defense in Depth**: If one model misses something, the other can catch it
- **Specialized Models**: Each model is optimized for its specific task

**Decision Logic:**
- If **either** Model 1 (email content) **OR** Model 2 (URLs) flags as phishing → **Final verdict: PHISHING**
- Uses the **maximum probability** from both models for confidence score
- This OR-based logic ensures we catch threats even if one model is uncertain

---

## Model 1: Email Content Analysis (DistilBERT)

### Architecture
- **Model Type**: DistilBERT (Distilled Bidirectional Encoder Representations from Transformers)
- **Framework**: PyTorch
- **Base Model**: `distilbert-base-uncased`
- **Task**: Binary classification (Phishing vs. Legitimate)
- **Accuracy**: ~92% on test set

### What Model 1 Analyzes

#### 1. **Text Content**
- **Subject Line**: Analyzes urgency, suspicious keywords, sender impersonation
- **Email Body**: Full text content including:
  - Grammatical errors and spelling mistakes
  - Urgency language ("act now", "limited time", "expires soon")
  - Suspicious requests (verify account, update payment, confirm identity)
  - Social engineering tactics

#### 2. **Semantic Understanding**
- **Context**: Understands meaning, not just keywords
  - Example: "Your account will be suspended" vs. "Account maintenance scheduled"
- **Sentiment**: Detects urgency, fear, and pressure tactics
- **Patterns**: Recognizes common phishing templates and structures

#### 3. **Preprocessing Pipeline**
```
Raw Email → Text Cleaning → Tokenization → Model Input
```

**Text Cleaning Steps:**
- HTML tag removal
- URL normalization (replaced with `[URL]` token)
- Email address anonymization (replaced with `[EMAIL]` token)
- Phone number anonymization (replaced with `[PHONE]` token)
- Whitespace normalization

**Tokenization:**
- Uses DistilBERT tokenizer
- Max length: 512 tokens
- Truncation for longer emails
- Padding for shorter emails

### Training Data
- **6 Datasets** totaling **76,926 emails**:
  - **Phishing**: CEAS_08, Nazario, Nigerian_Fraud (43,830 emails)
  - **Legitimate**: Enron, Ling, SpamAssassin (33,096 emails)
- **Class Balance**: 57% phishing, 43% legitimate (slightly imbalanced)
- **Preprocessing**: Combined subject + body for analysis

### Output
Returns a dictionary with:
- `is_phishing`: Boolean (True if phishing probability > 0.5)
- `phishing_probability`: Float (0.0 to 1.0)
- `legitimate_probability`: Float (0.0 to 1.0)
- `confidence`: Float (maximum of the two probabilities)

---

## Model 2: URL Detection (CNN)

### Architecture
- **Model Type**: Convolutional Neural Network (CNN)
- **Framework**: TensorFlow/Keras
- **Input**: Character-level tokenization of URLs
- **Task**: Binary classification (Phishing vs. Legitimate URLs)
- **Accuracy**: ~98% on test set

### What Model 2 Analyzes

#### 1. **URL Structure**
- **Domain Name**: Suspicious domains, typosquatting (e.g., "paypa1.com" vs "paypal.com")
- **Subdomain Patterns**: Unusual subdomain structures
- **Path Length**: Suspiciously long paths
- **Query Parameters**: Suspicious parameters, tracking IDs

#### 2. **Character-Level Patterns**
- **Character Sequences**: Recognizes patterns in character combinations
- **Special Characters**: Unusual use of hyphens, numbers, special chars
- **Length**: URL length patterns (phishing URLs often longer)
- **Encoding**: URL encoding patterns

#### 3. **Preprocessing Pipeline**
```
Raw URL → Character Tokenization → Padding/Truncation → Model Input
```

**Character Tokenization:**
- Each character becomes a token
- Max length: 200 characters
- Padding for shorter URLs
- Truncation for longer URLs

**Architecture Details:**
- **Embedding Layer**: 50-dimensional character embeddings
- **Convolutional Layers**: 3 Conv1D layers (kernel sizes: 3, 4, 5)
- **Filters**: 128 filters per layer
- **Pooling**: GlobalMaxPooling1D
- **Dense Layers**: 64-unit hidden layer + 2-unit output layer
- **Dropout**: 0.5 for regularization
- **Total Parameters**: ~179K (very lightweight!)

### Training Data
- **5 Datasets** from GitHub:
  - Benign URLs: ~28,532
  - Phishing URLs: ~22,977
  - **Total**: ~51,509 URLs in test set
- **Class Balance**: Approximately balanced

### Output
Returns a dictionary with:
- `is_phishing`: Boolean
- `phishing_probability`: Float (0.0 to 1.0)
- `legitimate_probability`: Float (0.0 to 1.0)
- `confidence`: Float

---

## Combined Detection Logic

### Decision Process

1. **Email Content Analysis** (Model 1):
   - Analyzes entire email text (subject + body)
   - Returns phishing probability: `P_email`

2. **URL Analysis** (Model 2):
   - Analyzes each URL found in the email
   - Returns phishing probability for each URL: `P_url1, P_url2, ...`
   - Takes **maximum** URL probability: `P_url_max = max(P_url1, P_url2, ...)`

3. **Final Decision**:
   ```python
   is_phishing = (P_email > 0.5) OR (P_url_max > 0.5)
   combined_confidence = max(P_email, P_url_max)
   ```

### Why This Works

- **Email-only phishing**: Model 1 catches phishing emails with no URLs (social engineering)
- **URL-only phishing**: Model 2 catches legitimate-looking emails with malicious links
- **Combined threats**: Both models agree → higher confidence
- **Edge cases**: If one model is uncertain, the other can still catch it

---

## Edge Cases and Limitations

### Edge Cases Handled

1. **Emails with No URLs**
   - Model 1 still analyzes content
   - Can detect phishing based on text alone
   - Example: "Call this number immediately" scams

2. **Emails with Multiple URLs**
   - Model 2 analyzes each URL separately
   - Takes maximum probability (if any URL is phishing, email is flagged)
   - Example: Legitimate email with one malicious link

3. **HTML/Encoded Emails**
   - System sends raw content to Model 1
   - Model 1's preprocessing handles HTML tags
   - Can analyze base64/quoted-printable encoded content

4. **Very Long Emails**
   - Model 1 truncates to 512 tokens
   - Analyzes first portion (most important content usually at top)
   - Still effective due to BERT's attention mechanism

5. **Very Short Emails**
   - Model 1 pads to 512 tokens
   - Can still detect based on keywords and structure
   - Example: "Verify account: [link]" → high phishing probability

6. **Legitimate Emails with Suspicious Keywords**
   - Model 1 uses context, not just keywords
   - "Urgent" in legitimate business context vs. phishing context
   - Semantic understanding prevents false positives

### Limitations

1. **Language**: Trained primarily on English emails
   - May have lower accuracy for non-English content
   - Solution: Could retrain with multilingual data

2. **New Attack Patterns**: 
   - May miss novel phishing techniques not in training data
   - Solution: Regular retraining with new data

3. **False Positives**:
   - Legitimate marketing emails with urgency language
   - Solution: User can review and provide feedback

4. **Model Confidence**:
   - Low confidence (30-70%) may indicate uncertainty
   - Solution: System shows confidence score to user

5. **URL Shorteners**:
   - Model 2 can't analyze destination until resolved
   - Solution: Could add URL expansion step (future enhancement)

---

## Technical Implementation Details

### API Server (Flask)
- **Lazy Loading**: Models load only when first prediction is requested
- **Memory Optimization**: Reduces startup memory footprint
- **Error Handling**: Graceful degradation if one model fails
- **Thread Safety**: Uses locks for concurrent requests

### Chrome Extension
- **Real-time Analysis**: Sends content to API server
- **Content Extraction**: Handles raw emails, HTML pages, Gmail/Outlook
- **Visual Feedback**: Shows probabilities, indicators, highlighted content
- **Explainability**: "Show More Details" explains why email was flagged

### Performance
- **Model 1 Inference**: ~100-200ms per email (GPU), ~500ms (CPU)
- **Model 2 Inference**: ~10-50ms per URL (very fast, lightweight model)
- **Combined**: Typically < 1 second for full analysis

---

## Explainability Features

### Indicators Shown to User

1. **Suspicious Keywords**: Highlights words like "urgent", "verify", "suspended"
2. **URL Analysis**: Shows phishing probability for each URL
3. **Confidence Score**: Overall threat score (0-100%)
4. **Sender Analysis**: Checks for suspicious sender patterns
5. **Grammar/Spelling**: Flags poor grammar (common in phishing)
6. **Urgency Detection**: Identifies pressure tactics
7. **URL Count**: Multiple URLs can be suspicious

### Visual Feedback
- **Color Coding**: Green (safe), Yellow (warning), Red (danger)
- **Probability Bars**: Visual representation of confidence
- **Highlighted Content**: Shows what the AI analyzed
- **Detailed Indicators**: Explains each red flag

---

## Training and Evaluation

### Model 1 Training
- **Epochs**: 3 (with early stopping)
- **Batch Size**: 16 (train), 32 (eval)
- **Learning Rate**: 2e-5
- **Optimizer**: AdamW
- **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- **Validation**: 20% of training data held out

### Model 2 Training
- **Epochs**: 30
- **Batch Size**: 1000
- **Optimizer**: Adam
- **Mixed Precision**: FP16 for faster training (GPU)
- **XLA Compilation**: TensorFlow optimization
- **Metrics**: Accuracy, Precision, Recall, F1-Score

### Evaluation Results
- **Model 1**: ~92% accuracy, balanced precision/recall
- **Model 2**: ~98% accuracy, excellent performance
- **Combined**: Higher accuracy when both models agree

---

## Future Enhancements

1. **URL Expansion**: Resolve shortened URLs before analysis
2. **Image Analysis**: OCR to detect phishing in email images
3. **Sender Reputation**: Check sender domain reputation
4. **Real-time Learning**: Update models with user feedback
5. **Multilingual Support**: Train on non-English emails
6. **Attachment Analysis**: Scan attachments for threats

---

## Conclusion

PhishGuard uses a sophisticated dual-model approach that combines:
- **Semantic understanding** (Model 1) for email content
- **Pattern recognition** (Model 2) for URL structure
- **Defense in depth** through OR-based decision logic
- **Explainability** through detailed indicators

This architecture provides robust phishing detection while maintaining transparency for users.

