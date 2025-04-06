# Advancing Phishing Email Detection: A Comparative Study of Deep Learning Models
Authors: Najwa Altwaijry, Isra Al-Turaiki, Reem Alotaibi, and Fatimah Alakeel

Published On: March 24th, 2024

Link: [read the full paper](https://pubmed.ncbi.nlm.nih.gov/38610289/)

Summary by: Zi Xuan Li

## Problem Statement:
This academic paper addresses the contemporary issue of detecting phishing in emails.
For reference, phishing is a cybersecurity threat that exploits social engineering tactics to deceive
individuals into revealing sensitive information such as login credentials, financial details, and
personal information. The paper outlines that cybercriminals often disguise phishing emails to
appear as legitimate communications from trusted sources often leading to high false positive
rates and limited adaptability in traditional phishing detection methods.
## Dataset Used:
Two datasets were used for the model’s evaluation:
1) [Phishing Corpus](https://academictorrents.com/details/a77cda9a9d89a60dbdfbe581adf6e2df9197995a)
2) [Spam Assassin](https://spamassassin.apache.org/old/publiccorpus/20050311_spam_2.tar.bz2)
## Methodologies Used:
The research employs deep-learning based techniques to detect phishing emails,
primarily focusing on one-dimensional convolutional neural networks (1D-CNNPD). To enhance
performance, the base 1D-CNNPD model was augmented with recurrent layers such as LSTM,
Bi-LSTM, GRU, and BI-GRU. The methodology includes multiple stages consisting of: data
preprocessing, model design, training, and evaluation.
  1) Data preprocessing (Data preparation): Phishing emails contain a mix of text, urls, and
metadata which require pre processing to extract meaningful features.
  2) Model design: A CNN based model was chosen because of its ability to progressively
extract text features at different levels to capture the context of the email (text data has
different levels of structure, low level features include individual letters and character
sequences, mid-level features include common words and phrases used in phishing, and
high-level features such as intent and sentence structure in phishing emails.)
  3) Model training: The models were trained on 70% of the dataset, validated on 30% of the
training set, and evaluated on the remaining 30% of the data set. (If there were 10,000
emails in the dataset: 7,000 emails are assigned to the training set, which is further split
into 4,900 for training and 2,100 for validation. The remaining 3,000 emails are used to
evaluate the model)
  4) Model evaluation: The model’s effectiveness is then assessed based on accuracy,
precision, recall, F1-score, and Receiver Operating Characteristics - Area Under Curve
score.
## Findings and Contributions:
Adapting a CNN-based model with recurrent layers improves phishing detection
performance. The 1D-CNNPD model with Bi-GRU achieved the best results. The study found
that increasing model depth initially improves performance but eventually leads to diminishing
returns and overfitting. The study highlights the potential of using lightweight deep learning
models for cybersecurity problems designed to be efficient in terms of computation, memory
storage, and inference speed instead of their alternative heavyweight models which have more
parameters, higher memory and power consumption, and slower inference time.
## Relevance to Project:
The findings of this paper align with the intentions for our machine learning phishing
detection application. The paper demonstrated the effectiveness of CNNs with recurrent layers
which suggests us to incorporate similar deep learning techniques into our project. Additionally,
the paper provides insights on data preparation and model training strategies.
## Citations:
Altwaijry, N.; Al-Turaiki, I.; Alotaibi, R.; Alakeel, F. Advancing Phishing Email Detection: A Comparative Study of Deep Learning Models. Sensors 2024, 24, 2077. https://doi.org/10.3390/s24072077


