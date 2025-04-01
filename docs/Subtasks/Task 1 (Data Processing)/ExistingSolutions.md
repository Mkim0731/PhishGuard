# Task 1: Data Processing – Research on Existing Technologies and Methodologies

## Goal
Prepare raw email data (body, headers, metadata, URLs) for feature extraction and modeling by:
- Cleaning HTML, scripts, and encoded text
- Normalizing formats
- Parsing headers and metadata
- Labeling phishing vs legitimate emails

---

## 🔧 Existing Technologies & Methodologies

| Tool/Library | Description | Strengths | Weaknesses | Applicability |
|-------------|-------------|-----------|------------|---------------|
| **BeautifulSoup** | Python library for parsing HTML/XML | Handles messy email bodies, easy to extract visible text | Struggles with malformed HTML; slower on large datasets | Good for cleaning up email bodies and extracting readable content |
| **Email Parser (`email` module in Python stdlib)** | Built-in Python tool to parse `.eml` files | Handles headers, MIME, attachments well | Can be verbose and lacks high-level abstraction | Good for parsing structure, headers, and multipart content |
| **Pandas** | Data manipulation library | Efficient for tabular handling, merging data from multiple sources | Not built for raw text cleaning directly | Useful for storing and organizing processed data, and merging metadata |
| **scikit-learn’s `LabelEncoder`** or manual labeling | Tools to encode phishing/ham labels | Simple and integrated with ML pipeline | Needs a clean dataset or manual labeling efforts | Useful once labels are acquired from sources like Kaggle or domain heuristics |

---

## 📚 References
- [BeautifulSoup Documentation](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)
- [Python Email Module Docs](https://docs.python.org/3/library/email.html)
- [Kaggle Phishing Datasets](https://www.kaggle.com/datasets/search?search=phishing+emails)
- Research Paper: **“Detecting Phishing Emails using Natural Language Processing”** (IEEE, 2021) – highlights preprocessing steps like HTML stripping and MIME parsing before tokenization.

---

## 📈 Effectiveness Evaluation

- **BeautifulSoup** is great for removing tags, scripts, and decoding text content. It boosts model performance by focusing on the core message.
- The `email` library is extremely accurate in parsing multipart emails and extracting technical header data (e.g., SPF, DKIM), crucial for phishing detection.
- **Limitations**:
  - Tools like BeautifulSoup don't handle encoding or attachments well.
  - The `email` library can be tedious when processing large datasets.
  - Labeling is a bottleneck if data lacks ground truth.

---

## 💡 Potential Enhancements / Ideas
- Use **`html2text`** or **`trafilatura`** for faster and more robust HTML to plain text conversion.
- Implement **heuristic-based pre-labeling** using sender domain, presence of suspicious links, and header analysis.
- Consider **distributed data processing** with Dask or PySpark if email volume is high.

