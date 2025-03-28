# Dataset Collection and Analysis
This section includes five datasets collected from various sources. Each dataset is described with key features, preprocessing steps, and any challenges that may be faced. We also provide a comparative analysis to justify the selection of datasets based on their relevance and quality. 
## Datasets

### Phishing Email Data by Type
- **Source:** [Kaggle](https://www.kaggle.com/datasets/charlottehall/phishing-email-data-by-type)
- **Description:** This dataset contains phishing, fraud, commercial spam, and legitimate emails. There are two features namely “Subject” & “Text” with a label titled “Type”.
- **Usability:**
    - Perform text cleaning: remove irrelevant symbols, numbers, and stop words (commonly used words such as “and” & “the”) to focus on meaningful features.
    - Feature extraction: convert the email content into structured data allowing for a machine learning algorithm to digest it.
- **Comparative Analysis:** Contains raw email content along with email subject and the label “Type” is already pre-labeled. 

### Phishing Email Detection
- **Source:** [Kaggle](https://www.kaggle.com/datasets/subhajournal/phishingemails)
- **Description:** This dataset contains both phishing and legitimate email samples, with its only feature being “Email Text” and only label being “Email Type”.
- **Usability:**
    - Perform text cleaning: remove irrelevant symbols, numbers, and stop words (commonly used words such as “and” & “the”) to focus on meaningful features.
    - Feature extraction: convert the email content into structured data allowing for a machine learning algorithm to digest it.
    - Balancing: Preventing biased learning where the model over predicts the majority label. 
- **Comparative Analysis:** Contains raw email content and the label “Email Type” is already pre-labeled and can be easily encoded.
  
### PhiUSIIL Phishing URL (Website)
- **Source:** [UC Irvine Machine Learning Repository](https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset)
- **Description:** This dataset contains over 235,000 URL instances with multiple extracted features and is particularly useful for more extensive experiments. Some of those URLs are legit and some are phishing/spam.
- **Usability:**
    - Having a list of safe and non-safe URLs will help us perform URL/domain name scanning before even having to look at the contents of the email sent.
    - We can make this dataset dynamic where we add to it email/domain address that our users have safely dealt with before.
- **Comparative Analysis:**
    - Label 1 corresponds to a legitimate URL, label 0 to a phishing URL. Also, it includes a lot of other attributes for a given address.
      
### Ling-Spam Dataset
- **Source:** [Kaggle](https://www.kaggle.com/datasets/mandygu/lingspam-dataset)
- **Description:** Contains a collection of phishing and legitimate emails. Includes single feature “Email Text” and a label “Email Type” to indicate if an email is phishing or legitimate
- **Usability:**
    - Clean the email content by removing unnecessary symbols, numbers and common stop words to emphasize important features
    - Structure the raw email content into a format suitable for machine learning algorithms to process
    - Address class imbalance to avoid the model favoring the majority class (non - phishing)
- **Comparative Analysis:** The dataset provides raw email content along with pre-labeled email types, making it suitable for binary classification. However, compared to other datasets, it offers fewer features, which may require additional preprocessing to enhance the model's performance.
  
### Phishing Email Dataset
- **Source:** [Kaggle](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset?utm_source=chatgpt.comDatasets)
- **Description:** This dataset contains phishing and legitimate email samples. It includes both the subject and body of emails with a corresponding label indicating if the email is phishing or not.
- **Usability:**
    - Perform comprehensive text cleaning, removing special characters, numbers, and stop words to focus on the core message.
    - Feature extraction on both subject and body text to capture contextual information for classification.
- **Comparative Analysis:** With both subject and body included, this dataset allows for deeper analysis and potentially more accurate phishing detection models. It supports binary classification tasks and can be directly used in natural language processing (NLP) pipelines

## Citations
Hall, Charlotte. Phishing Email Data by Type. Kaggle, 2023, <br>
https://www.kaggle.com/datasets/charlottehall/phishing-email-data-by-type.

SubhaJournal. Phishing Emails. Kaggle, 2023, <br>
https://www.kaggle.com/datasets/subhajournal/phishingemails.

Zhang, Xiyue. PHIUSIIL Phishing URL Dataset. UCI Machine Learning Repository, 2023, <br>
https://archive.ics.uci.edu/dataset/967/phiusiil+phishing+url+dataset.

Gu, Mandy. LingSpam Dataset. Kaggle, 2023,<br>
https://www.kaggle.com/datasets/mandygu/lingspam-dataset.

Abdullah Alam, Naser. Phishing Email Dataset. Kaggle, 2023, <br> 
https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset?utm_source=chatgpt.comDatasets
