# PhishGuard: Presentation Talking Points

## Quick Answer (30 seconds)

"PhishGuard uses two complementary AI models working together. **Model 1** analyzes email content using DistilBERT to understand semantic meaning and detect phishing language patterns. **Model 2** analyzes URLs using a CNN to detect suspicious URL structures. If either model flags something as phishing, we mark it as a threat. This dual approach catches both content-based phishing and malicious links."

---

## Detailed Explanation (2-3 minutes)

### How It Works

**1. Email Content Analysis (Model 1 - DistilBERT)**
- Analyzes the **entire email text** (subject + body)
- Uses **semantic understanding** - not just keyword matching
- Looks for:
  - Urgency language ("act now", "expires soon")
  - Suspicious requests ("verify account", "update payment")
  - Grammar/spelling errors (common in phishing)
  - Social engineering patterns
- **Accuracy**: ~92% on test data
- **Training**: 76,926 emails from 6 datasets

**2. URL Analysis (Model 2 - CNN)**
- Analyzes **each URL** found in the email
- Uses **character-level patterns** to detect:
  - Typosquatting (e.g., "paypa1.com" vs "paypal.com")
  - Suspicious domain structures
  - Unusual URL patterns
- **Accuracy**: ~98% on test data
- **Training**: ~51,000 URLs (phishing + legitimate)

**3. Combined Decision**
- If **either** model says phishing → **Final verdict: PHISHING**
- Uses **maximum probability** from both models for confidence
- This OR-based logic ensures we catch threats even if one model is uncertain

### What It Analyzes

**Email Content:**
- Subject line urgency and suspicious keywords
- Body text semantic meaning
- Grammar and spelling quality
- Social engineering tactics
- Overall email structure and patterns

**URLs:**
- Domain name authenticity
- Character-level patterns
- URL structure and length
- Query parameters
- Encoding patterns

### Edge Cases Handled

1. **Emails with no URLs**: Model 1 still analyzes content
2. **Multiple URLs**: Model 2 checks each, flags if any are suspicious
3. **HTML/Encoded emails**: System handles base64, quoted-printable encoding
4. **Very long emails**: Truncates to 512 tokens (most important content at top)
5. **Very short emails**: Can still detect based on keywords and structure
6. **Legitimate emails with suspicious words**: Uses context, not just keywords

### Limitations

- Primarily trained on English emails
- May miss novel attack patterns not in training data
- False positives possible with legitimate marketing emails
- URL shorteners require expansion (future enhancement)

---

## Technical Highlights

### Model 1 (DistilBERT)
- **Architecture**: Transformer-based (BERT variant)
- **Framework**: PyTorch
- **Size**: ~66M parameters
- **Inference**: ~100-200ms (GPU), ~500ms (CPU)

### Model 2 (CNN)
- **Architecture**: Convolutional Neural Network
- **Framework**: TensorFlow/Keras
- **Size**: ~179K parameters (very lightweight!)
- **Inference**: ~10-50ms per URL

### System Features
- **Lazy Loading**: Models load only when needed (saves memory)
- **Real-time Analysis**: < 1 second for full analysis
- **Explainability**: Shows why email was flagged (keywords, URLs, indicators)
- **Visual Feedback**: Color-coded threat levels, probability bars

---

## Key Points for Professor

1. **Dual-Model Defense**: Two specialized models provide comprehensive coverage
2. **Semantic Understanding**: Not just keyword matching - understands context
3. **High Accuracy**: 92% (email) + 98% (URL) = robust detection
4. **Explainable AI**: Users can see why emails were flagged
5. **Production-Ready**: Optimized for real-world deployment
6. **Well-Trained**: Large, diverse training datasets (76K+ emails, 51K+ URLs)

---

## Example Scenarios

**Scenario 1: Phishing Email with Malicious Link**
- Model 1: Detects suspicious content ("urgent", "verify account")
- Model 2: Detects malicious URL (typosquatting)
- **Result**: Both agree → High confidence phishing

**Scenario 2: Legitimate Email with Suspicious Keywords**
- Model 1: Uses context to understand it's legitimate business communication
- Model 2: URLs are legitimate
- **Result**: Both agree → Legitimate

**Scenario 3: Well-Written Phishing Email**
- Model 1: May be uncertain (good grammar, professional tone)
- Model 2: Detects malicious URL
- **Result**: Model 2 catches it → Flagged as phishing

---

## Questions You Might Get

**Q: Why two models instead of one?**
A: Email content and URLs provide different signals. A single model would need to learn both, reducing specialization. Two models allow each to excel at its task.

**Q: What if both models disagree?**
A: We use OR logic - if either says phishing, we flag it. This is more conservative and catches more threats, though it may increase false positives slightly.

**Q: How do you handle new attack patterns?**
A: The models are trained on diverse datasets, but novel attacks may be missed. Solution: Regular retraining with new data, and user feedback for continuous improvement.

**Q: What about false positives?**
A: We show confidence scores and detailed indicators so users can make informed decisions. Low confidence (30-70%) suggests uncertainty.

**Q: Can it detect phishing in images?**
A: Currently no - only text and URLs. Future enhancement could add OCR for image analysis.

---

## Conclusion

PhishGuard demonstrates a practical application of modern AI (transformers + CNNs) to solve a real-world cybersecurity problem. The dual-model approach provides robust, explainable phishing detection suitable for end-user deployment.

