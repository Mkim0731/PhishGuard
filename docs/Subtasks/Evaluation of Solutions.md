# Content Analysis

### Existing Solutions:
* Gmail’s Spam & Phishing Detection System: Multi-layered system with NLP, template matching, and metadata analysis. Highly scalable, trained on billions of emails.
* BERT/RoBERTa: High accuracy but resource-heavy
### Limitations:
* Struggles with adversarial text (misspellings, Unicode tricks)
* No unified model for text + metadata
### Improvements:
* Hybrid model: DistilBERT (lightweight) + XGBoost (structured features)
  * DistilBERT: Lightweight transformer for semantic analysis (60% faster than BERT).
  * XGBoost: Handles structured features (e.g., sender history, grammar errors).
  * Combination: Merge model outputs with meta-learner for final score.
* Adversarial training with generated phishing examples.
  * Train models to flag these evasive techniques.

# Link Checks

### Existing Solutions:
* Regex/Heuristics: Fast but brittle
* URLNet: Detects obfuscation but data-hungry
### Limitations:
* Fails against zero-day domains
* No real-time URL expansion checks
### Improvements:
* Live DNS/WHOIS lookup + screenshot analysis
  * Flag domains registered <7 days ago.
  * Detect mismatched registrar/country.

# Spoof Detection

### Existing Solutions:
* Python email/dkimpy: Protocol-compliant but low-level
### Limitations:
* SPF/DKIM can be spoofed
* No behavioral analysis (e.g., sudden geo changes)
### Improvements:
* Add sender reputation scoring (past emails, domain age) based on:
  * Historical sending patterns.
  * Domain age + TLS adoption.
  * Geographic consistency (e.g., login attempts vs. email origin).

# Real Time Alerts

### Existing Solutions:
* Chrome Extensions: Tight UI integration
### Limitations:
* Gmail DOM changes break functionality
* Alerts lack context
### Improvements:
* DOM-agnostic API-based detection (Gmail Add-on)
  * Avoid UI breakage on updates.
  * Access raw headers/metadata reliably.


