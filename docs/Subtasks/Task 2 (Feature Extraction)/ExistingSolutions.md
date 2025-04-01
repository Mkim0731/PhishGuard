# Task 2: Feature Extraction – Research on Existing Technologies and Methodologies

## Goal
Transform cleaned email data into structured numerical features that can be used for machine learning. This includes:
- Text-based features (e.g., word usage, sentiment)
- URL and domain analysis
- Email header & metadata features
- Embeddings or vectorization methods

---

## 🔧 Existing Technologies & Methodologies

| Tool/Library | Description | Strengths | Weaknesses | Applicability |
|-------------|-------------|-----------|------------|---------------|
| **TF-IDF (via scikit-learn)** | Transforms text into weighted word vectors | Simple, effective for sparse email content | Doesn’t capture context or semantics | Solid baseline for textual features |
| **Word2Vec / GloVe** | Word embeddings trained on large corpora | Captures semantic similarity | Needs a lot of data; static embeddings | Useful for building context-aware text classifiers |
| **URLNet / PhishNet** | Deep learning models for URL-based phishing detection | Handles obfuscated URLs and character patterns | Requires large labeled datasets; more complex to train | Advanced option for analyzing suspicious URLs |
| **Header Feature Extraction (Custom)** | Extract SPF, DKIM, DMARC, IPs, time anomalies | Directly captures phishing-specific patterns | No off-the-shelf tool; requires manual coding | High relevance for phishing detection models |

---

## 📚 References
- [Scikit-learn TF-IDF Docs](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [Mikolov et al., 2013 – Word2Vec](https://arxiv.org/abs/1301.3781)
- [URLNet: Learning a URL Representation with Deep Learning for Malicious URL Detection (2018)](https://arxiv.org/abs/1802.03162)
- [PhishNet](https://www.sciencedirect.com/science/article/pii/S266628172200018X)
- Blog: [Understanding DKIM, SPF, and DMARC](https://postmarkapp.com/guides/spf-dkim-dmarc)

---

## 📈 Effectiveness Evaluation

- **TF-IDF** is efficient and widely used in spam/phishing detection with solid baseline accuracy.
- **Embeddings like Word2Vec** offer better semantic understanding but require large corpora.
- **URLNet** is cutting-edge for detecting obfuscated malicious links, outperforming regex-based detection.
- **Header-based features** are phishing-specific, but implementation is manual.

---

## 💡 Potential Enhancements / Ideas
- Combine **TF-IDF + header features** for lightweight, interpretable models.
- Use **pre-trained embeddings** like BERT for more nuanced understanding of email content.
- Create a **custom phishing URL score** using domain age, registrar info, and character frequency patterns.
- Develop a feature union pipeline using `FeatureUnion` (scikit-learn) for combining diverse features into a single vector.

