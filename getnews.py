import requests
import pandas as pd
import numpy as np
import datetime as dt
import time
from dateutil.relativedelta import relativedelta
import re
import nltk
from data_cleaning_processing.find_missing_dates import find_missing_dates
import yfinance as yf

nltk.download("stopwords")
nltk.download('wordnet')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


def get_news(company, date_from, news_num=1000):
    header = {"YOUR_API_KEY_HERE"}
    params = {"ticker": company,
              "limit": news_num,
              "order": "desc",
              "published_utc.lte": date_from
              }

    get_news_url = 'https://api.polygon.io/v2/reference/news'
    response = requests.get(url=get_news_url, headers=header, params=params)

    if response.status_code == 200:
        print("Request was successful!")
    else:
        print("Request failed with status code:", response.status_code)

    news_response = response.json()
    news = news_response["results"]

    data_list = []
    for new_id in range(len(news)):
        data = {}
        news_title = news[new_id]["title"]
        news_title = news_title.replace(",", "")
        date = news[new_id]['published_utc'][:10]
        last_day = news[-1]["published_utc"]
        data["Date"] = date
        data["News_Title"] = news_title
        try:
            insights = news[new_id]["insights"]
            for dic in insights:
                if dic["ticker"] == company:
                    sentiment_val = dic["sentiment"]
                    sentiment_reasoning = dic["sentiment_reasoning"]
                    if sentiment_val == "positive":
                        sentiment = 1
                    elif sentiment_val == "neutral":
                        sentiment = 0
                    else:
                        sentiment = -1
                    data["Sentiment_Reasoning"] = sentiment_reasoning
                    data["Sentiment"] = sentiment
        except KeyError:
            data["Sentiment"] = ""
            data["Sentiment_Reasoning"] = ""
        data_list.append(data)

    df = pd.DataFrame(data_list)
    return last_day, df


def combine_news(all_targets):
    todays_date = dt.datetime.now()
    todays_date = todays_date.strftime('%Y-%m-%d')
    todays_date_completed = todays_date + "T" + str(dt.datetime.now().time())[:8] + "Z"
    upto_date = dt.datetime.now() - relativedelta(years=4)
    upto_date = upto_date.date()

    for comp in all_targets:
        company = comp
        df_combined = pd.DataFrame()
        news = get_news(comp, date_from=todays_date_completed)
        last_day = news[0]
        df = news[1]
        get_date = last_day[:10]
        get_date = dt.datetime.strptime(get_date, "%Y-%m-%d").date()
        df_combined = pd.concat([df_combined, df])

        while upto_date < get_date:
            time.sleep(15)
            older_news = get_news(comp, date_from=last_day)
            new_last_day = older_news[0]
            older_date_df = older_news[1]
            df_combined = pd.concat([df_combined, older_date_df])
            get_date = new_last_day[:10]
            get_date = dt.datetime.strptime(get_date, "%Y-%m-%d").date()
            last_day = new_last_day
            print(get_date)

        df_combined.sort_values("Date", ascending=False, inplace=True)
        try:
            df_combined.to_csv(f"target_company_news/{company}_news.csv", encoding='utf-8', index=False)
        except:
            df_combined.to_csv(f"company_news/target_company_news/{company}_news.csv",
                               encoding='utf-8', index=False)

        print(f"{company} news saved!")


def rate_news(all_targets):
    for company in all_targets:

        try:
            target_news = pd.read_csv(f"target_company_news/{company}_news.csv", encoding="utf-8")
        except:
            target_news = pd.read_csv(f"company_news/target_company_news/{company}_news.csv",
                                      encoding="utf-8")
        target_news.drop_duplicates(subset=["News_Title"], inplace=True)
        target_news = target_news.loc[target_news.Sentiment != 0]

        target_ratings = target_news.Sentiment.to_list()
        target_valid_ratings = [sentiment for sentiment in target_ratings if sentiment in [-1, 1]]
        target_ratings_num = len(target_valid_ratings)

        target_company_df = target_news.iloc[:target_ratings_num, :]
        target_company_sent_reasonings = target_company_df.Sentiment_Reasoning.to_list()

        try:
            example_news_df = pd.read_csv("all_big_company_news.csv",
                                          encoding="utf-8")
        except:
            example_news_df = pd.read_csv("company_news/all_big_company_news.csv",
                                          encoding="utf-8")
        example_news_df.drop_duplicates(subset=["News_Title"], inplace=True)
        example_news_df = example_news_df.loc[example_news_df.Sentiment != 0]

        combined_list = [example_news_df, target_news]
        combined_df = pd.concat(combined_list)
        combined_df.drop_duplicates(inplace=True)
        combined_df.replace('', np.nan, inplace=True)
        combined_df.dropna(subset=['Sentiment'], inplace=True)
        combined_df.sort_values("Date", ascending=False, inplace=True)

        stop_words = set(stopwords.words("english"))

        stop_words_list = sorted(list(stop_words))

        words_to_remove = [
            "but", "not", "again", "no", "nor", "very", "should",
            "just", "only", "really", "and", "or", "if", "while",
            "never", "none", "some", "many", "few", "like", "as", "such", "doesn't",
            "isn't", "aren't"
        ]

        for word in words_to_remove:
            if word in stop_words_list:
                stop_words.remove(word)

        lemmatizer = WordNetLemmatizer()
        cleaned_text = []

        news_titles = list(combined_df.News_Title)
        sentiment_reasonings = list(combined_df.Sentiment_Reasoning)

        for i in range(combined_df.shape[0]):
            new_title_text = re.sub("[^a-zA-Z]", " ", news_titles[i]).lower()
            if not pd.isna(sentiment_reasonings[i]):
                sentiment_text = re.sub("[^a-zA-Z]", " ", sentiment_reasonings[i]).lower()
                review = new_title_text + " " + sentiment_text
            else:
                review = new_title_text
            review = review.split()

            cleaned_text.append(" ".join(review))

        cv = CountVectorizer(max_features=4441)
        X = cv.fit_transform(cleaned_text).toarray()
        # print(len(X[0]))

        df_bow = pd.DataFrame(X, columns=cv.get_feature_names_out())
        X = df_bow.to_numpy()
        y = combined_df.iloc[:, -1].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=5)

        encoder = LabelEncoder()
        y_train = encoder.fit_transform(y_train)
        y_test = encoder.transform(y_test)

        classifier = LogisticRegression(solver="lbfgs", penalty="l2", C=1).fit(X_train, y_train)

        #K-Fold Cross Validation
        # kf = KFold(n_splits=100, shuffle=True, random_state=42)
        # scores = cross_val_score(classifier, X, y, cv=kf)
        #
        # print("Cross-validation scores:", scores)
        # print("Mean score:", scores.mean())

        y_pred = classifier.predict(X_test)

        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        acc_score = accuracy_score(y_test, y_pred)
        print(acc_score)

        # error analysis for the wrong predictions
        original_text = cv.inverse_transform(X_test)
        x_test = [" ".join(text) for text in original_text]
        yhat = list(y_pred)
        actual_y = list(y_test)
        indexs = [x for x in range(len(actual_y)) if actual_y[x] != yhat[x]]
        actual_y_filtered = []
        yhat_filtered = []
        news_filtered = []
        for index in indexs:
            actual_y_filtered.append(actual_y[index])
            yhat_filtered.append(yhat[index])
            news_filtered.append(x_test[index])

        dic = {"News": news_filtered,
               "Real Rating": actual_y_filtered,
               "Predicted Result": yhat_filtered}
        results = pd.DataFrame(dic)
        results.to_csv("error_analysis.csv", encoding="utf-8", index=False)

        #Predict the rest of the news not included and did not have Sentiment
        news_data_completed = []
        new_cleaned_text = []
        predict_df = target_news.iloc[target_ratings_num:, :]
        for x in range(predict_df.shape[0]):
            new_data = {}
            news_title = predict_df.iloc[x, 1]
            date = predict_df.iloc[x, 0]
            new_data["Date"] = date
            new_data["News_Title"] = news_title
            new_title_text = re.sub("[^a-zA-Z]", " ", news_title).lower().split()
            new_title_text = [lemmatizer.lemmatize(word) for word in new_title_text if word not in stop_words]
            new_cleaned_text.append(" ".join(new_title_text))
            news_data_completed.append(new_data)

        missing_sentiment_news = cv.transform(new_cleaned_text).toarray()
        get_new_sentiments = classifier.predict(missing_sentiment_news)

        #Format and add the predictions back to the filtered Dataset
        for sent in range(len(list(get_new_sentiments))):
            news_data_completed[sent]["Sentiment"] = list(get_new_sentiments)[sent]

        for c in range(target_company_df.shape[0]):
            old_data = {"Date": target_company_df.iloc[c, 0],
                        "News_Title": target_company_df.iloc[c, 1],
                        "Sentiment_Reasoning": target_company_sent_reasonings[c],
                        "Sentiment": target_company_df.iloc[c, 3]}
            news_data_completed.append(old_data)

        #Export combined DataFrames with the predicted Sentiments
        new_sentiment_data = pd.DataFrame(news_data_completed)
        new_sentiment_data.sort_values("Date", ascending=False, inplace=True)
        new_sentiment_data['Sentiment'] = new_sentiment_data['Sentiment'].replace(0.0, -1.0)
        try:
            new_sentiment_data.to_csv(f"ml_rated_news/{company}_rated_news.csv",
                                      encoding='utf-8', index=False)
        except:
            new_sentiment_data.to_csv(f"company_news/ml_rated_news/{company}_rated_news.csv",
                                      encoding='utf-8', index=False)


def count_sentiments(all_targets):
    for company in all_targets:
        #get rated news for each company in the list
        try:
            rated_news_df = pd.read_csv(f"ml_rated_news/{company}_rated_news.csv", encoding="utf-8")
        except:
            rated_news_df = pd.read_csv(f"company_news/ml_rated_news/{company}_rated_news.csv",
                                        encoding="utf-8")

        #get dates without repeat, last day and create iteration number
        last_day = rated_news_df.iloc[-1, 0]
        iter_num = rated_news_df["Date"].nunique()
        all_dates = list(rated_news_df["Date"].drop_duplicates())

        #filter through negative news and reformat it
        negative_news = rated_news_df.loc[rated_news_df.Sentiment == -3.0]
        count_negative_news_df = negative_news.groupby("Date").agg({"Sentiment": pd.Series.count})
        count_negative_news_df.sort_values("Date", ascending=False, inplace=True)

        reshaped_negatives = count_negative_news_df.values.reshape(1, -1)[0, :]
        negative_scores_list = reshaped_negatives.tolist()
        negative_dates_list = count_negative_news_df.index.tolist()

        # filter through positive news and reformat it
        positive_news = rated_news_df.loc[rated_news_df.Sentiment == 1.0]
        count_positive_news_df = positive_news.groupby("Date").agg({"Sentiment": pd.Series.count})
        count_positive_news_df.sort_values("Date", ascending=False, inplace=True)

        reshaped_positives = count_positive_news_df.values.reshape(1, -1)[0, :]
        positive_scores_list = reshaped_positives.tolist()
        positive_dates_list = count_positive_news_df.index.tolist()

        #filter through positive and negative news get all the scores add them to the list
        positive_scores = []
        negative_scores = []
        for i in range(len(all_dates)):
            date = all_dates[i]
            if date in positive_dates_list:
                get_positive_index = positive_dates_list.index(date)
                get_positive_score = positive_scores_list[get_positive_index]
                positive_scores.append(get_positive_score)
            else:
                get_positive_score = 0
                positive_scores.append(get_positive_score)

            if date in negative_dates_list:
                get_negative_index = negative_dates_list.index(date)
                get_negative_score = negative_scores_list[get_negative_index]
                negative_scores.append(get_negative_score)
            else:
                get_negative_score = 0
                negative_scores.append(get_negative_score)

        #minus positive news score from negative score to get overall score for the day
        overall_score = []
        for x in range(len(all_dates)):
            score = int(positive_scores[x]) - (int(negative_scores[x] * 3))
            overall_score.append(score)

        #create dictionary to make a DataFrame
        scored_dates_dic = {
            "Date": all_dates,
            "Overall_Score": overall_score
        }

        #Export Dictionary
        scored_dates_df = pd.DataFrame(scored_dates_dic)

        try:
            scored_dates_df.to_csv(f"date_n_scores/{company}_dates_scores.csv", encoding='utf-8', index=False)
        except:
            scored_dates_df.to_csv(f"company_news/date_n_scores/{company}_dates_scores.csv", encoding='utf-8',
                                   index=False)


def fix_dates(all_targets):
    for company in all_targets:

        try:
            ml_rated_news_df = pd.read_csv(f"date_n_scores/{company}_dates_scores.csv", encoding="utf-8")
        except:
            ml_rated_news_df = pd.read_csv(f"company_news/date_n_scores/{company}_dates_scores.csv",
                                           encoding="utf-8")

        news_dates = ml_rated_news_df.Date.tolist()
        missing_dates = find_missing_dates(news_dates)
        fill_scores = [0 for d in range(len(missing_dates))]
        missing_df = {"Date": missing_dates,
                      "Overall_Score": fill_scores
                      }
        missing_df = pd.DataFrame(missing_df)
        combined_df = pd.concat([ml_rated_news_df, missing_df])
        combined_df.sort_values("Date", ascending=False, inplace=True)

        try:
            combined_df.to_csv(f"date_n_scores/dates_fixed/{company}_dates_scores_fixed.csv", encoding="utf-8",
                               index=False)
        except:
            combined_df.to_csv(f"company_news/date_n_scores/dates_fixed/{company}_dates_scores_fixed.csv",
                               encoding="utf-8",
                               index=False)


def add_latest_news(all_companies):
    for company in all_companies:
        try:
            df = pd.read_csv(f"biggest_company_news/{company}_news.csv", encoding="utf-8")
        except:
            df = pd.read_csv(f"company_news/biggest_company_news/{company}_news.csv", encoding="utf-8")

        latest_date = df.iloc[0, 0]
        filter_date = "2019-02-27"

        todays_date = dt.datetime.now()
        todays_date = todays_date.strftime('%Y-%m-%d')
        todays_date_completed = todays_date + "T" + str(dt.datetime.now().time())[:8] + "Z"
        news = get_news(company, date_from=todays_date_completed)
        new_news = pd.DataFrame(news[1])
        filter_news = new_news.loc[new_news.Date > latest_date]

        combine_newest = pd.concat([df, filter_news])
        combine_newest.sort_values("Date", ascending=False, inplace=True)
        combine_newest = combine_newest.loc[combine_newest.Date > filter_date]

        try:
            combine_newest.to_csv(f"biggest_company_news/{company}_news.csv", encoding="utf-8", index=False)
        except:
            combine_newest.to_csv(f"company_news/biggest_company_news/{company}_news.csv", encoding="utf-8",
                                  index=False)

        if len(all_companies) > 1:
            time.sleep(15)


def combine_big_company_news(all_companies):
    all_sentiments = []
    all_sentiment_reasonings = []
    all_news_titles = []
    all_dates = []

    for company in all_companies:
        # Load data
        try:
            company_news = pd.read_csv(f"biggest_company_news/{company}_news.csv", encoding="utf-8")
        except:
            company_news = pd.read_csv(f"company_news/biggest_company_news/{company}_news.csv", encoding="utf-8")
        # company_news['Sentiment'] = company_news['Sentiment'].replace(0.0, -1.0)
        company_news = company_news.loc[company_news.Sentiment != 0]

        # Step 1: Filter valid sentiments
        company_ratings = company_news.Sentiment.to_list()

        company_valid_ratings = [sentiment for sentiment in company_ratings if sentiment in [-1, 1]]
        company_ratings_num = len(company_valid_ratings)

        # Step 2: Create the dataframe with filtered rows
        company_filtered_df = company_news.iloc[:company_ratings_num, :]

        #get all rows of data
        company_date = company_filtered_df.Date.to_list()
        company_news_title = company_filtered_df.News_Title.to_list()
        company_sentiment_reasoning = company_filtered_df.Sentiment_Reasoning.to_list()
        company_sentiment = company_filtered_df.Sentiment.to_list()

        for i in range(len(company_date)):
            all_dates.append(company_date[i])
            all_news_titles.append(company_news_title[i])
            all_sentiment_reasonings.append(company_sentiment_reasoning[i])
            all_sentiments.append(company_sentiment[i])

    #create a dictionary to put all the extracted data into it
    all_data = {"Date": all_dates,
                "News_Title": all_news_titles,
                "Sentiment_Reasoning": all_sentiment_reasonings,
                "Sentiment": all_sentiments
                }

    # news examples
    try:
        example_news_df = pd.read_csv("biggest_company_news/newsexamples_cleaned.csv",
                                      encoding="utf-8")
    except:
        example_news_df = pd.read_csv("company_news/biggest_company_news/newsexamples_cleaned.csv",
                                      encoding="utf-8")

    example_news_df = example_news_df.loc[example_news_df.Sentiment != 0]

    #combine news_examples and biggest company news together
    all_data = pd.DataFrame(all_data)
    all_news = pd.concat([example_news_df, all_data])
    all_news.sort_values("Date", ascending=False, inplace=True)
    all_news.drop_duplicates(inplace=True)
    all_news.replace('', np.nan, inplace=True)
    all_news.dropna(subset=['Sentiment'], inplace=True)
    try:
        all_news.to_csv("all_big_company_news.csv", encoding="utf-8", index=False)
    except:
        all_news.to_csv("company_news/all_big_company_news.csv", encoding="utf-8", index=False)


def merge_ets_news(targets):
    all_ets_news = []
    for c in targets:
        news_path = f"biggest_company_news/{c}_news.csv"
        try:
            df = pd.read_csv(news_path, low_memory=False)
            all_ets_news.append(df)
        except FileNotFoundError:
            pass
    if not all_ets_news:
        return None
    out = pd.concat(all_ets_news, ignore_index=True, sort=False)
    out.drop_duplicates(inplace=True)
    out.to_csv("target_company_news/etf_news.csv", index=False)
    print("etfs news are merged!")
    return "etf_news.csv"


biggest_companies = ["AAPL", "AMD", "AMZN", "BRK.A", "DIS", "F", "GOOG", "IBM", "INTC", "JNJ", "JPM", "KO", "MA", "MCD",
                     "META", "MSFT", "NFLX", "NVDA", "PEP", "PG", "SBUX", "TSLA", "V", "WMT", "XOM", "PLTR", "SPY",
                     "RSP", "QQQ", "IWM", "DIA", "VTI", "XLK", "XLF", "XLY", "TLT", "HYG"]

etfs = ["SPY", "RSP", "QQQ", "IWM", "DIA", "VTI", "XLK", "XLF", "XLY", "TLT", "HYG"]

target_companies = ["etf"]

##BIG COMPANY NEWS FUNCTIONS
# add_latest_news(biggest_companies)
combine_big_company_news(biggest_companies)

## SINGLE COMPANY NEWS FUNCTIONS
# combine_news(all_targets=etfs)
# rate_news(all_targets=target_companies)
# count_sentiments(all_targets=target_companies)
# fix_dates(all_targets=target_companies)

##ETF news
# merge_ets_news(etfs)
