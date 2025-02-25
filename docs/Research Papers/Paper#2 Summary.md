# Phishing Email Detection Model Using Deep Learning
Authors: Samer Atawneh and Hamzah Aljehani

Published On: October 15th, 2023

Link: [read the full paper](https://www.mdpi.com/2079-9292/12/20/4261)

Summary by: Peng Gao

## Problem Statement:
Phishing emails are a major cybersecurity threat, often used to deceive recipients into sharing
sensitive information like passwords, personal details, and financial data. Despite the growing
awareness, phishing attacks continue to be an effective method used by cybercriminals. The
main challenge lies in distinguishing between phishing emails and legitimate emails. The
paper’s goal is to develop an automated system that can accurately identify phishing emails with
minimal false positives.
## Dataset Used:
The authors used publicly available datasets that include both phishing and legitimate emails.
This is critical because the quality and diversity of data impacts the performance of the model.
Below are links to the datasets:
1) https://www.kaggle.com/datasets/rtatman/fraudulent-email-corpus
2) https://www.kaggle.com/datasets/maharshipandya/email-spam-dataset-extended
3) https://www.kaggle.com/datasets/charlottehall/phishing-email-data-by-type
4) https://www.kaggle.com/datasets/yashpaloswal/spamham-email-classification-nlp
## Methodologies Used:
The research utilized several advanced deep learning methodologies to build an efficient
phishing mail detection model. Several techniques and models applied are:
  - Convolutional Neural Networks(CNNs): Although initially used in image recognition,
CNNs have proven be to effective in text classification as well. This allows the model to
capture patterns or structures within the text.
  - Long Short-Term Memory (LSTM): LSTMs are a type of recurrent neural network
designed to address the vanishing gradient problem, allowing them to remember
long-range dependencies in sequences. This is useful for analyzing email text, where
the meaning often depends on the context of multiple words in a sequence.
  - Bidirectional Encoder Representations from Transformers (BERT): BERT is a pretrained
transformer model that process text in a single direction. BERT use directional approach
to understand context from both left and right simultaneously. This enhances the model’s
ability to understand the meaning of each word based on its surrounding words.
  - Recurrent Neural Networks(RNNs): RNNs are used to process sequences of words in
emails, where the model is capable of remembering previous words to predict the next
word. This is important for detecting phishing emails that have patterns or features that
appear across multiple words.
## Findings and Contributions:
- High Accuracy: The result of the study is the high accuracy achieved by the deep
learning models. The BERT and LSTM combination outperformed other models,
achieving an accuracy of 99.61% in detecting phishing emails.
- Model Comparison: The paper compared the performance of various deep learning
models, showing that CNNs, LSTMs, and RNNs are all effective but fall short compared
to BERT models. Suggesting that BERT’s ability to capture context in emails make it a
powerful approaching for phishing detection
## Relevance to Project:
- Deep Learning models: Using BERT combined with LSTM would provide a strong
foundation for the model in this project.
- Natural Language Processing (NLP): The paper also emphasizes the importance of NLP,
applying NLP methods to process email will be crucial for extracting features from
emails.
- Performance Benchmark: The high accuracy reported in the paper sets a benchmark
that can be aimed for to achieving similar performance.
## Citations:
Atawneh, S.; Aljehani, H. Phishing Email Detection Model Using Deep Learning. Electronics 2023, 12, 4261. https://doi.org/10.3390/electronics12204261
