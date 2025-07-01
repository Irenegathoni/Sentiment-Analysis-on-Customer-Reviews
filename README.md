# Sentiment Analysis on Customer Reviews

## Overview

This project focuses on building a sentiment analysis system that classifies customer reviews as either positive or negative. The solution leverages natural language processing (NLP) techniques for text preprocessing, feature extraction, model training, and visualization. Additionally, the project includes a deployed Streamlit web application where users can input text and receive real-time sentiment predictions.

## Problem Statement

In today's digital economy, businesses rely heavily on customer feedback. Analyzing sentiments at scale can be tedious and error-prone if done manually. This project aims to automate sentiment classification using machine learning, offering both analytical insights and an interactive prediction tool.

## Objectives

- Clean and preprocess raw text data effectively.
- Train a machine learning model to classify text sentiments.
- Visualize key patterns and top contributing words.
- Deploy a user-friendly interface for real-time sentiment prediction.

## Dataset

The dataset used is a collection of customer reviews labeled with their respective sentiments.  
- Format: CSV  
- Columns: `Text`, `Sentiment`  
- Size: Approximately 3000+ records  
- Source: Synthetic and open-source review datasets

## Tools & Technologies

- **Programming Language:** Python 3.10+
- **Libraries:**  
  - Text Processing: `re`, `string`, `spaCy`  
  - Machine Learning: `scikit-learn`, `pandas`, `numpy`  
  - Visualization: `matplotlib`, `seaborn`, `wordcloud`  
  - Web App: `Streamlit`  
- **Model Persistence:** `pickle`
## Workflow

1. **Preprocessing** (`01_preprocessing.ipynb`):
   - Lowercasing, removing URLs, digits, punctuation, and whitespace.
   - Tokenization and stopword removal using spaCy.
   - Output: cleaned CSV file.

2. **Modeling** (`02_sentiment_model.ipynb`):
   - TF-IDF vectorization.
   - Model training using Logistic Regression.
   - Evaluation using accuracy score and confusion matrix.
   - Saved both vectorizer and model with `pickle`.

3. **Visual Analysis** (`03_summary_visuals.ipynb`):
   - Word frequency analysis by sentiment class.
   - Word clouds for positive and negative reviews.
   - Bar charts of top tokens per sentiment.

4. **App Deployment** (`app/sentiment_app.py`):
   - Built with Streamlit.
   - User inputs text, and the app returns predicted sentiment and confidence level.



