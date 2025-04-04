# URL & Link Analysis

## Goal
Scan embedded links in emails for signs of phishing or malicious behavior.

## Existing Technologies & Methodologies

### 1. URLNet
- Deep learning model that treats URLs as sequences of characters and learns patterns.

**Strengths:**
- Detects obfuscated URLs.
- Learns from raw data without manual feature engineering.

**Weaknesses:**
- Needs a lot of labeled training data.
- More complex to train and deploy.

**GitHub:** [URLNet](https://github.com/antonyt/urlNet)

---

### 2. Regex / Heuristic-Based Analysis
- Use of pattern matching for blacklisted terms, domain features.

**Strengths:**
- Fast and simple to implement.
- Can flag known suspicious patterns quickly.

**Weaknesses:**
- Easy to bypass with small mutations.
- High false positives.

---

### 3. PhishTank / VirusTotal API
- Query external threat intelligence databases to verify URLs.

**Strengths:**
- Provides up-to-date data on malicious domains.
- Easy to integrate.

**Weaknesses:**
- Depends on third-party availability and API rate limits.

**APIs:** [PhishTank](https://www.phishtank.com/api_info.php) | [VirusTotal](https://developers.virustotal.com/)
