# How PhishGuard Scoring Works - Detailed Explanation

## Overview

PhishGuard uses a **dual-model scoring system** that combines email content analysis and URL analysis to produce a final **Threat Score** (0-100%). The scoring process involves multiple steps, adjustments, and a weighted combination to provide accurate and nuanced threat assessment.

---

## Step-by-Step Scoring Process

### Step 1: Model 1 - Email Content Probability

**What Happens:**
1. Email text is sent to Model 1 (DistilBERT)
2. Model tokenizes the text (converts to numbers)
3. Model processes through neural network layers
4. Model outputs **logits** (raw scores for each class)
5. Logits are converted to probabilities using **softmax**

**Mathematical Process:**
```python
# Inside Model 1's predict() method:
encoding = tokenizer(email_text)  # Convert text to numbers
outputs = model(**encoding)       # Run through DistilBERT
logits = outputs.logits          # Get raw scores

# Softmax converts logits to probabilities (sum to 1.0)
probabilities = softmax(logits)

# Extract probabilities
phishing_prob = probabilities[0][1]      # Class 1 (phishing)
legitimate_prob = probabilities[0][0]   # Class 0 (legitimate)

# Example output:
# phishing_prob = 0.75 (75%)
# legitimate_prob = 0.25 (25%)
# Sum = 1.0 (100%)
```

**Example:**
- Input: "Subject: URGENT! Verify your account now. Click here: http://..."
- Model 1 Output: `phishing_probability = 0.80` (80%), `legitimate_probability = 0.20` (20%)

---

### Step 2: Keyword Adjustment (Reduces False Positives)

**Purpose**: Prevent single suspicious words from causing high false positives

**Suspicious Keywords List:**
- 'urgent', 'immediately', 'verify', 'suspended', 'expired'
- 'click here', 'act now', 'confirm', 'update'
- 'verify account', 'account locked', 'payment required'
- 'prize winner', 'congratulations', 'free money'
- And more...

**Adjustment Logic:**

```python
# Count suspicious keywords in email
found_keywords = count_keywords_in_email(email_text)

if keyword_count == 1:
    # Only 1 keyword found → Cap at 25% phishing
    phishing_prob = min(phishing_prob, 0.25)
    legitimate_prob = 1.0 - phishing_prob  # Recalculate to sum to 1.0
    
elif keyword_count == 0:
    # No keywords → Reduce by 10% (slightly favor legitimate)
    phishing_prob = phishing_prob * 0.9
    legitimate_prob = 1.0 - phishing_prob
    
else:
    # 2+ keywords → Use full model prediction (no adjustment)
    # phishing_prob stays as-is
```

**Examples:**

**Example A: Single Keyword**
- Model 1 predicts: 60% phishing
- Only "urgent" found
- **After adjustment: 25% phishing** (capped), 75% legitimate

**Example B: Multiple Keywords**
- Model 1 predicts: 75% phishing
- Found: "urgent", "verify", "suspended" (3 keywords)
- **After adjustment: 75% phishing** (no cap, multiple indicators)

**Example C: No Keywords**
- Model 1 predicts: 30% phishing
- No suspicious keywords found
- **After adjustment: 27% phishing** (reduced by 10%), 73% legitimate

---

### Step 3: Model 2 - URL Probability

**What Happens:**
1. Each URL is sent to Model 2 (CNN)
2. URL is converted to character sequences
3. CNN processes through convolutional layers
4. Model outputs probabilities for each URL

**Mathematical Process:**
```python
# Inside Model 2's predict_url() method:
seq = tokenizer.texts_to_sequences([url])  # Convert URL to numbers
padded = pad_sequences(seq, maxlen=200)    # Pad/truncate to 200 chars
predictions = model.predict(padded)       # Run through CNN

# Extract probabilities
legitimate_prob = predictions[0][0]  # Class 0 (legitimate)
phishing_prob = predictions[0][1]    # Class 1 (phishing)

# Example output for one URL:
# phishing_probability = 0.85 (85%)
# legitimate_probability = 0.15 (15%)
```

**Multiple URLs:**
- If email has multiple URLs, Model 2 analyzes each one
- Each URL gets its own probability score
- Example:
  - URL 1: 85% phishing
  - URL 2: 60% phishing
  - URL 3: 20% phishing

---

### Step 4: Combined Decision (`is_phishing` Flag)

**Purpose**: Determine if email should be marked as phishing

**Logic:**
```python
email_prob = email_result['phishing_probability']  # After keyword adjustment
url_probs = [url['phishing_probability'] for url in url_results]

# Email threshold: > 65%
email_suspicious = email_prob > 0.65

# URL threshold: > 70% AND at least 2 URLs (or 1 URL > 85%)
high_risk_urls = [p for p in url_probs if p > 0.70]
url_suspicious = len(high_risk_urls) >= 2 or (len(high_risk_urls) >= 1 and max(url_probs) > 0.85)

# Very high individual scores
very_high_email = email_prob > 0.85
very_high_url = max(url_probs) > 0.90 if url_probs else False

# Mark as phishing if:
is_phishing = (email_suspicious AND url_suspicious) OR very_high_email OR very_high_url
```

**Decision Table:**

| Email Prob | URL Prob | URL Count | Result |
|------------|----------|-----------|--------|
| 70% | 75% | 2+ URLs | ✅ Phishing (both suspicious) |
| 90% | 20% | 1 URL | ✅ Phishing (very high email) |
| 60% | 95% | 1 URL | ✅ Phishing (very high URL) |
| 60% | 50% | 1 URL | ❌ Not Phishing (neither high enough) |
| 70% | 65% | 1 URL | ❌ Not Phishing (URL not high enough) |

---

### Step 5: Threat Score Calculation (Weighted Average)

**Purpose**: Combine email and URL probabilities into single threat score (0-100%)

**Formula:**
```
Threat Score = (Email_Probability × 60%) + (URL_Probability × 40%)
```

**URL Probability Calculation:**
- **If multiple URLs**: Average of top 2 URLs
- **If single URL**: That URL's probability
- **If no URLs**: 0% (only email component used)

**Implementation:**
```javascript
// In popup.js displayResults() function:

const emailProb = result.email_result.phishing_probability;  // e.g., 0.75
const urlProbs = result.url_results.map(r => r.phishing_probability);  // e.g., [0.85, 0.60]

let urlProb = 0;

if (urlProbs.length >= 2) {
    // Average of top 2 URLs
    const top2 = urlProbs.sort((a, b) => b - a).slice(0, 2);
    urlProb = (top2[0] + top2[1]) / 2;  // e.g., (0.85 + 0.60) / 2 = 0.725
} else if (urlProbs.length === 1) {
    urlProb = urlProbs[0];  // e.g., 0.85
} else {
    urlProb = 0;  // No URLs
}

// Weighted average
const threatScore = (emailProb * 0.6) + (urlProb * 0.4);
// e.g., (0.75 × 0.6) + (0.725 × 0.4) = 0.45 + 0.29 = 0.74 = 74%
```

**Why Weighted Average?**
- **Email content is more reliable** (60% weight) - provides context and meaning
- **URLs can be misleading** (40% weight) - legitimate sites may have suspicious-looking URLs
- **More nuanced than max** - considers both indicators proportionally
- **Prevents single indicator dominance** - requires both to contribute

---

## Complete Example Walkthrough

### Example Email:
```
Subject: URGENT: Verify Your Account Immediately
Body: Your account has been suspended. Click here to verify:
http://fake-bank.com/verify?token=abc123
https://bit.ly/redirect123
```

### Step 1: Model 1 Analysis
- **Input**: Full email text
- **Model 1 Output**: 
  - Raw prediction: 80% phishing, 20% legitimate

### Step 2: Keyword Adjustment
- **Keywords found**: "urgent", "verify", "suspended", "immediately" (4 keywords)
- **Adjustment**: No cap (2+ keywords)
- **Final Email Probability**: 80% phishing, 20% legitimate

### Step 3: Model 2 Analysis
- **URL 1**: `http://fake-bank.com/verify?token=abc123`
  - Model 2 Output: 85% phishing
- **URL 2**: `https://bit.ly/redirect123`
  - Model 2 Output: 70% phishing
- **URL Probabilities**: [0.85, 0.70]

### Step 4: Combined Decision
- **Email**: 80% (> 65%) ✓ Suspicious
- **URLs**: 2 URLs with > 70% (85% and 70%) ✓ Suspicious
- **Result**: `is_phishing = TRUE` (both suspicious)

### Step 5: Threat Score Calculation
- **Email Probability**: 0.80 (80%)
- **URL Probability**: Average of top 2 = (0.85 + 0.70) / 2 = 0.775 (77.5%)
- **Threat Score**: (0.80 × 0.6) + (0.775 × 0.4) = 0.48 + 0.31 = **79%**

### Step 6: Display
- **Threat Score**: 79% (displayed in circle)
- **Color**: Red (Danger - 79% ≥ 75%)
- **Warning Box**: Shows (79% > 50%)
- **Email Probabilities**: 80% phishing, 20% legitimate
- **URL Scores**: 85% and 70% (both shown)

---

## Another Example: Single Keyword Case

### Example Email:
```
Subject: Urgent action required
Body: Please respond to this email as soon as possible.
```

### Step 1: Model 1 Analysis
- **Model 1 Output**: 60% phishing, 40% legitimate

### Step 2: Keyword Adjustment
- **Keywords found**: "urgent" (1 keyword only)
- **Adjustment**: Cap at 25%
- **Final Email Probability**: **25% phishing** (capped), 75% legitimate

### Step 3: Model 2 Analysis
- **No URLs found**
- **URL Probability**: 0%

### Step 4: Combined Decision
- **Email**: 25% (< 65%) ✗ Not suspicious
- **URLs**: None
- **Result**: `is_phishing = FALSE`

### Step 5: Threat Score Calculation
- **Email Probability**: 0.25 (25%)
- **URL Probability**: 0% (no URLs)
- **Threat Score**: (0.25 × 0.6) + (0 × 0.4) = 0.15 + 0 = **15%**

### Step 6: Display
- **Threat Score**: 15%
- **Color**: Green (Safe - 15% < 40%)
- **Warning Box**: Hidden (15% ≤ 50%)
- **Email Probabilities**: 25% phishing, 75% legitimate

---

## Key Scoring Principles

### 1. Probabilities Always Sum to 100%
- `phishing_probability + legitimate_probability = 1.0`
- If one is 75%, the other is 25%

### 2. Keyword Adjustment Prevents False Positives
- Single word like "urgent" → Max 25% (not enough evidence)
- Multiple indicators → Full model prediction (stronger evidence)

### 3. Weighted Average Provides Balance
- Email (60%) + URLs (40%) = More accurate than just max
- Considers both indicators proportionally

### 4. High Thresholds Reduce False Positives
- Email: 65% threshold (was 50%)
- URL: 70% threshold (was 50%)
- Combined: Requires both OR very high individual

### 5. Multiple URLs Averaged
- Top 2 URLs averaged (prevents single bad URL from dominating)
- If only 1 URL, uses that URL's probability

---

## Scoring Formula Summary

```
FINAL THREAT SCORE = (Email_Prob × 0.6) + (URL_Prob × 0.4)

Where:
- Email_Prob = Model1 prediction (after keyword adjustment)
- URL_Prob = Average of top 2 URLs (or single URL, or 0 if none)
- Result is multiplied by 100 to get percentage (0-100%)
```

---

## Visual Flow Diagram

```
Email Text
    ↓
[Model 1: DistilBERT]
    ↓
Raw Probability (e.g., 60%)
    ↓
[Keyword Check]
    ├─ 1 keyword → Cap at 25%
    ├─ 0 keywords → Reduce by 10%
    └─ 2+ keywords → Use full prediction
    ↓
Adjusted Email Probability (e.g., 25% or 60% or 75%)
    ↓
                    ┌─────────────────┐
                    │  Weighted Avg   │
                    │  (Email × 60%)  │
                    └─────────────────┘
                            +
                    ┌─────────────────┐
URLs → [Model 2: CNN] → URL Probabilities
    ↓                    ↓
Average of Top 2    │  Weighted Avg   │
(e.g., 70%)         │  (URL × 40%)   │
                    └─────────────────┘
                            ↓
                    FINAL THREAT SCORE
                    (e.g., 45% or 73%)
```

---

This scoring system ensures:
- ✅ **Accurate**: Uses state-of-the-art ML models
- ✅ **Nuanced**: Provides detailed probabilities, not just yes/no
- ✅ **Balanced**: Considers both email and URLs proportionally
- ✅ **Robust**: Prevents false positives from single indicators
- ✅ **Explainable**: Shows users why something is flagged

