Advancing Phishing Email Detection: A Comparative Study of Deep Learning Models
Authors: N. Altwaijry, I. Al-Turaiki, R. Alotaibi, F. Alakeel
Published On: March 24, 2024
Journal: Sensors, Volume 24, Article 2077
Academic Editors: He Fang and Sherali Zeadally
Link: https://doi.org/10.3390/s24072077
Summary by: Minsu Kim

Problem Statement:
The academic paper addresses the escalating threat of phishing emails, a major vector for cyberattacks that continues to evolve in complexity. Traditional detection methods often struggle to keep up with the dynamic nature of phishing tactics, leading to increased vulnerabilities. The authors aim to enhance phishing email detection accuracy by evaluating and comparing various deep learning architectures, focusing on the integration of convolutional and recurrent neural network models.

Dataset Used:
The study utilizes two widely recognized datasets for evaluation:

Phishing Corpus: A collection of phishing emails used to train and test detection algorithms.
Spam Assassin: A dataset of legitimate emails that helps in creating a balanced training environment and in minimizing false positives.
Methodologies Used:
The researchers developed and compared several deep learning models for phishing email detection:

1D Convolutional Neural Network for Phishing Detection (1D-CNNPD): A baseline model designed to process textual email data using convolutional layers.
Enhanced Hybrid Models:
1D-CNNPD + LSTM: Integrating Long Short-Term Memory layers to capture sequential dependencies.
1D-CNNPD + Bi-LSTM: Using Bidirectional LSTM for better contextual understanding.
1D-CNNPD + GRU: Employing Gated Recurrent Units for efficient sequence learning.
1D-CNNPD + Bi-GRU: A bidirectional approach using GRUs for improved accuracy.
The study emphasizes the fusion of convolutional and recurrent layers to leverage both feature extraction and sequential data analysis.

Findings and Contributions:
The hybrid model 1D-CNNPD + Bi-GRU achieved the best performance with:
Precision: 100%
Accuracy: 99.68%
F1 Score: 99.66%
Recall: 99.32%
The study demonstrated that combining CNN with Bi-GRU effectively captures both local features and contextual relationships in email text, leading to higher detection rates.
It highlights the importance of hybrid architectures in enhancing the robustness and reliability of phishing email detection systems.
Relevance to Project:
The findings of this study are highly relevant to the PhishGuard project. The demonstrated success of hybrid deep learning models provides a solid foundation for enhancing the detection capabilities of our system. Integrating Bi-GRU with CNN layers could significantly improve accuracy, helping PhishGuard effectively identify sophisticated phishing attempts. Moreover, the paper’s use of standard datasets allows for reproducibility, enabling us to benchmark our results against the study’s outcomes.

Citation:
Altwaijry, N.; Al-Turaiki, I.; Alotaibi, R.; Alakeel, F. Advancing Phishing Email Detection: A Comparative Study of Deep Learning Models. Sensors 2024, 24, 2077. https://doi.org/10.3390/s24072077.
