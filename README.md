

# Amazon Alexa Review Sentiment Analysis

This repository contains a project focused on sentiment analysis of Amazon Alexa product reviews. The goal of this project is to classify customer reviews as positive or negative using natural language processing (NLP) and machine learning techniques.

## Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Installation](#installation)
- [Data Preprocessing](#data-preprocessing)
- [Modeling](#modeling)
- [Evaluation](#evaluation)
- [Results](#results)
- [Visualization](#visualization)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

The project involves analyzing customer reviews of Amazon Alexa products to determine the sentiment behind each review. The sentiment analysis is performed using various machine learning models, and the best-performing model is identified based on evaluation metrics.

## Dataset

The dataset used for this project is the "Amazon Alexa Reviews" dataset, which includes several features such as:

- `rating`: The rating given by the customer.
- `date`: The date of the review.
- `variation`: The variation of the Alexa product.
- `verified_reviews`: The text of the customer review.
- `feedback`: The binary sentiment label (1 for positive, 0 for negative).

The dataset can be obtained from [Kaggle](https://www.kaggle.com/sid321axn/amazon-alexa-reviews).

## Installation

To run this project, you'll need to have Python installed along with the necessary libraries. You can install the required dependencies using pip:

```bash
git clone https://github.com/your-username/amazon-alexa-review-sentiment-analysis.git
cd amazon-alexa-review-sentiment-analysis
pip install numpy pandas scikit-learn nltk matplotlib seaborn wordcloud
```

## Data Preprocessing

Before feeding the data into machine learning models, several preprocessing steps are performed:

1. **Text Cleaning**: The review texts are cleaned by removing special characters, numbers, and stop words.
2. **Tokenization**: The text is tokenized into individual words.
3. **Lemmatization**: The words are lemmatized to their base form to reduce dimensionality.
4. **Vectorization**: The text data is converted into numerical form using techniques such as TF-IDF or Count Vectorizer.

## Modeling

Various machine learning models are applied to the processed data to classify the sentiment of the reviews:

- **Logistic Regression**: A simple and interpretable model for binary classification.
- **Random Forest Classifier**: An ensemble method that builds multiple decision trees and merges them to get a more accurate and stable prediction.
- **Support Vector Machine (SVM)**: A powerful model for classification tasks that tries to find the optimal hyperplane to separate the classes.
- **Naive Bayes**: A probabilistic model based on Bayes' theorem, particularly effective for text classification.

Additionally, model tuning is performed using GridSearchCV to find the best hyperparameters for each model.

## Evaluation

The performance of the models is evaluated using various metrics:

- **Accuracy**: The percentage of correctly predicted reviews out of the total reviews.
- **Precision, Recall, and F1-Score**: Metrics that evaluate the performance of the classification models, especially in cases of imbalanced datasets.
- **Confusion Matrix**: A table that describes the performance of the classification model on a set of test data for which the true values are known.

## Results

The results section presents the performance metrics for each model, highlighting the best-performing model based on accuracy, precision, recall, and F1-score. The model is then used to predict sentiments on a separate test set to validate its performance.

## Visualization

The project includes visualizations to help understand the distribution of sentiments, the importance of features, and the results of the classification:

- **WordCloud**: Visualization of the most frequent words in positive and negative reviews.
- **Bar Charts**: To display the distribution of sentiments and model performance metrics.
- **Confusion Matrix Heatmap**: A visual representation of the confusion matrix.



## Contributing

Contributions are welcome! If you have suggestions or improvements, feel free to submit a Pull Request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


