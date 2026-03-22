## Stock News Sentiment ML

Predicts whether financial news headlines are positive or negative using machine learning. Includes a full data scraping pipeline that collects labelled news from the Polygon.io API across 37 major tickers and ETFs.
Dataset

31,000+ labelled news headlines scraped from Polygon.io API
Covers major stocks (AAPL, NVDA, TSLA, GOOG, META, etc.) and ETFs (SPY, QQQ, etc.)
Labels: 1 (positive), -1 (negative), sourced from Polygon's built in sentiment insights
Date range: mid 2024 to March 2026
Class distribution: ~72% positive, ~28% negative
Zero nulls, fully cleaned

## Data Pipeline
The scraping system in getnews.py does the following:

- Pulls up to 1000 news articles per API call from Polygon.io with pagination
- Loops backwards through time with rate limiting (15s between calls) to collect up to 4 years of history per ticker
- Extracts headline, date, sentiment label and sentiment reasoning for each article
- Deduplicates and cleans the data
- Aggregates daily sentiment scores per ticker, weighting negative news 3x heavier than positive
- Fills in missing dates (weekends, holidays) with neutral scores
- Exports everything to structured CSVs

## Baseline Model

## CountVectorizer + Logistic Regression

- Bag of words approach using scikit learn CountVectorizer (max 4441 features)
- Text preprocessing: regex cleaning, lowercasing, stop word removal (kept sentiment words like "not", "never", "no"), lemmatization
- Combined news title + sentiment reasoning as input text
- Logistic Regression with L2 penalty
- 80/20 train test split
- Accuracy: ~96%

## VADER (Rule Based Baseline)

- NLTK SentimentIntensityAnalyzer applied to News_Title only (no Sentiment_Reasoning, to keep it fair)
- Compound score threshold: >= 0.05 = positive, < 0.05 = negative
- Accuracy: 61%
- Precision: 0.44 (negative), 0.83 (positive)
- Recall: 0.78 (negative), 0.53 (positive)
- VADER struggles with financial language because it was built for social media text, not news headlines

  ## RoBERTa (Pretrained Transformer)

- cardiffnlp/twitter-roberta-base-sentiment applied to News_Title only
- Predicted positive if positive score > negative score
- Accuracy: 83%
- Precision: 0.70 (negative), 0.91 (positive)
- Recall: 0.82 (negative), 0.83 (positive)
- Massive improvement over VADER, but still trained on Twitter data not financial news
  
## LSTM (Trained on Financial Headlines)

- Keras Sequential model: Embedding (GloVe 100d, trainable) + LSTM(128) + Dropout(0.3) + Dense(1, sigmoid)
- Trained on News_Title only, preprocessed with lowercase and special character removal, no stop word removal
- Padded sequences to maxlen=20 (matching actual headline length, not an arbitrary 100)
- Class weights applied to handle 72/28 imbalance
- Early stopping with patience=3
- Accuracy: 89%
- Precision: 0.88 (negative), 0.89 (positive)
- Recall: 0.75 (negative), 0.95 (positive)
- Beat RoBERTa because it learned financial language specifically rather than generalising from Twitter

Goal is to compare all approaches on the same dataset and document which performs best on financial news specifically.
## Model Comparison

| Model | Accuracy | Type | Input |
|-------|----------|------|-------|
| Logistic Regression | 96% | Trained | Title + Reasoning |
| LSTM | 89% | Trained | Title only |
| RoBERTa | 83% | Pretrained | Title only |
| VADER | 61% | Rule based | Title only |

Project Structure

getnews.py - full scraping and data pipeline
all_big_company_news.csv - combined labelled dataset
data_cleaning_processing/ - helper scripts for date fixing and data preprocessing
biggest_company_news/ - individual ticker news CSVs
target_company_news/ - target company specific news
date_n_scores/ - daily sentiment scores per ticker
ml_rated_news/ - model rated news outputs
error_analysis.csv - misclassification analysis
vader_approach.ipynb - VADER sentiment analysis and evaluation
roberta_approach.ipynb - RoBERTa transformer sentiment analysis and evaluation
vader_confusion_matrix.png - VADER confusion matrix visualisation
roberta_confusion_matrix.png - RoBERTa confusion matrix visualisation
LSTM_approach.ipynb - LSTM model training and evaluation
lstm_confusion_matrix.png - LSTM confusion matrix visualisation

## Built With

Python, pandas, NumPy, scikit learn, NLTK, requests
Polygon.io API for news data

## What I Learned

- Building a full data pipeline from API to model, not just downloading a Kaggle CSV
- Rate limiting and pagination for large scale API scraping
- Stop word removal needs to be done carefully for sentiment tasks, removing words like "not" destroys meaning
- Weighting negative sentiment heavier than positive gives more useful daily scores for financial analysis
- 96% accuracy sounds great but the class imbalance (72/28 split) means the model could be leaning on the majority class, worth investigating further with precision/recall per class
- Rule based sentiment models like VADER perform poorly on financial text (61%) because financial language uses different patterns than social media
- Pretrained transformer models like RoBERTa (83%) massively outperform rule based models like VADER (61%) on financial text, even when neither was trained specifically on financial data
- The gap between RoBERTa (83%) and the logistic regression baseline (96%) is partly because the baseline used sentiment reasoning text which essentially leaks the answer
- A trained LSTM (89%) outperformed a pretrained transformer (RoBERTa 83%) on domain specific financial text because it learned the patterns of financial headlines directly rather than relying on general social media language
- Padding sequence length matters hugely, setting maxlen=100 when headlines are only 10-15 words meant the LSTM was reading 90% zeros and could not learn, dropping to maxlen=20 fixed it immediately (67% to 89%)
- Class imbalance must be handled explicitly, without class weights the LSTM just predicted the majority class for every input
