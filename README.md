Stock News Sentiment ML

Predicts whether financial news headlines are positive or negative using machine learning. Includes a full data scraping pipeline that collects labelled news from the Polygon.io API across 37 major tickers and ETFs.
Dataset

31,000+ labelled news headlines scraped from Polygon.io API
Covers major stocks (AAPL, NVDA, TSLA, GOOG, META, etc.) and ETFs (SPY, QQQ, etc.)
Labels: 1 (positive), -1 (negative), sourced from Polygon's built in sentiment insights
Date range: mid 2024 to March 2026
Class distribution: ~72% positive, ~28% negative
Zero nulls, fully cleaned

Data Pipeline
The scraping system in getnews.py does the following:

Pulls up to 1000 news articles per API call from Polygon.io with pagination
Loops backwards through time with rate limiting (15s between calls) to collect up to 4 years of history per ticker
Extracts headline, date, sentiment label and sentiment reasoning for each article
Deduplicates and cleans the data
Aggregates daily sentiment scores per ticker, weighting negative news 3x heavier than positive
Fills in missing dates (weekends, holidays) with neutral scores
Exports everything to structured CSVs

Baseline Model
**CountVectorizer + Logistic Regression**

Bag of words approach using scikit learn CountVectorizer (max 4441 features)
Text preprocessing: regex cleaning, lowercasing, stop word removal (kept sentiment words like "not", "never", "no"), lemmatization
Combined news title + sentiment reasoning as input text
Logistic Regression with L2 penalty
80/20 train test split
Accuracy: ~96%

## VADER (Rule Based Baseline)

- NLTK SentimentIntensityAnalyzer applied to News_Title only (no Sentiment_Reasoning, to keep it fair)
- Compound score threshold: >= 0.05 = positive, < 0.05 = negative
- Accuracy: 61%
- Precision: 0.44 (negative), 0.83 (positive)
- Recall: 0.78 (negative), 0.53 (positive)
- VADER struggles with financial language because it was built for social media text, not news headlines

Planned Models

VADER (rule based sentiment baseline)
RoBERTa (pretrained transformer)
LSTM (sequential deep learning approach)

Goal is to compare all approaches on the same dataset and document which performs best on financial news specifically.
Project Structure

getnews.py - full scraping and data pipeline
all_big_company_news.csv - combined labelled dataset
data_cleaning_processing/ - helper scripts for date fixing and data preprocessing
biggest_company_news/ - individual ticker news CSVs
target_company_news/ - target company specific news
date_n_scores/ - daily sentiment scores per ticker
ml_rated_news/ - model rated news outputs
error_analysis.csv - misclassification analysis

Built With

Python, pandas, NumPy, scikit learn, NLTK, requests
Polygon.io API for news data

What I Learned

Building a full data pipeline from API to model, not just downloading a Kaggle CSV
Rate limiting and pagination for large scale API scraping
Stop word removal needs to be done carefully for sentiment tasks, removing words like "not" destroys meaning
Weighting negative sentiment heavier than positive gives more useful daily scores for financial analysis
96% accuracy sounds great but the class imbalance (72/28 split) means the model could be leaning on the majority class, worth investigating further with precision/recall per class
Rule based sentiment models like VADER perform poorly on financial text (61%) because financial language uses different patterns than social media
