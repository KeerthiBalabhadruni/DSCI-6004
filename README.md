# DSCI-6004 Sentiment Analysis of Amazon Customer Reviews using Transformers

This project utilizes Transformers, a state-of-the-art natural language processing (NLP) library, to perform sentiment analysis on Amazon customer reviews. The goal is to classify the sentiment of the reviews as positive, negative, or neutral.
### Overview
Sentiment analysis is a subfield of NLP that aims to determine the emotional tone behind a piece of text. In this project, we focus on analyzing the sentiment of customer reviews from Amazon. By utilizing Transformer-based models, specifically pre-trained models like BERT, GPT, or RoBERTa, we can leverage their contextual understanding of language to achieve accurate sentiment classification.

### Datasets:
The dataset utilized in this project is sourced from Kaggle and comprises Amazon Fine Food Reviews. It encompasses a comprehensive collection of customer reviews spanning various food products available on the Amazon platform. This dataset offers a rich repository of textual data, including reviews, ratings, timestamps, and helpfulness votes, providing a holistic view of customer sentiments and preferences.The Amazon Fine Food Reviews dataset consists of structured data fields such as ID, Product ID, User ID, Profile Name, Helpfulness Numerator, Helpfulness Denominator, Score, Time, Summary, and Text. These attributes offer valuable insights into the characteristics of each review, enabling detailed analysis and sentiment classification.

https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews?resource=download

### Requirements
  Python 3.x
  
  PyTorch
  
  Transformers
  
  Pandas
  
  NumPy
  
  Matplotlib
  
  Seaborn

### Usage
Data Preparation: Obtain Amazon customer review data. This can be done by scraping Amazon's website or using datasets available on platforms like Kaggle.

Preprocessing: Preprocess the data by cleaning, tokenizing, and encoding it into a format suitable for input into the Transformer model.

Model Selection: Choose a pre-trained Transformer model suitable for sentiment analysis. Common choices include BERT, GPT, RoBERTa, etc.

Fine-Tuning: Fine-tune the selected model on the Amazon customer review dataset. This involves training the model on the labeled data to adapt it to the specific task of sentiment classification.

Evaluation: Evaluate the fine-tuned model on a separate validation set to assess its performance. Common evaluation metrics include accuracy, precision, recall, and F1-score.

Inference: Use the trained model to perform sentiment analysis on new, unseen Amazon customer reviews.

### Conclusion
In conclusion, our sentiment analysis of Amazon customer reviews using transformer-based models and rule-based approaches has provided valuable insights into customer sentiments towards various products. Through rigorous data preprocessing, model training, and evaluation, we have successfully classified sentiments as positive, negative,or neutral, enabling businesses to better understand customer preferences and experiences. Our analysis revealed the effectiveness of transformer-based models, particularly Roberta, in accurately classifying sentiments across a diverse range of reviews. These models outperformed traditional rule-based approaches like VADER, demonstrating the superiority of deep learning based techniques in capturing complex language semantics. By visualizing the results and comparing model performance using various evaluation metrics, we have equipped businesses with actionable insights to improve product offerings, enhance marketing strategies, and optimize customer engagement. The diverse distribution of sentiments across reviews underscores the importance of tailored approaches to address customer feedback effectively.
