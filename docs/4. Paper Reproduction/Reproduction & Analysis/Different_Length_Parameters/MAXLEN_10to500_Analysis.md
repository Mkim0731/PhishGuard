Recommended MAX_LEN = 200

Reason:

Highest Test Accuracy: 0.9735

Benign Precision/Recall: 0.97 / 0.98

Phishing Precision/Recall: 0.98 / 0.96

Balanced Confusion Matrix:

True Negative (TN): 28083

False Positive (FP): 449

False Negative (FN): 917

True Positive (TP): 22060

Even though MAX_LEN=300 and 500 also achieved 0.9735 accuracy, MAX_LEN=200 is preferred because:

Training/inference time is shorter (smaller sequence size)

No measurable accuracy benefit from larger input length beyond 200

Computational efficiency matters (especially for real-time phishing detection)
