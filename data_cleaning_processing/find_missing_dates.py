from datetime import datetime, timedelta
import yfinance as yf


def create_dates_list(time_period_days):
    a_year = []
    while time_period_days > 0:
        end_date = datetime.now()
        first_day = end_date.strftime('%Y-%m-%d')
        one_day_back = end_date - timedelta(days=time_period_days)
        if time_period_days == 365:
            a_year.append(first_day)
        time_period_days -= 1
        a_year.append(one_day_back.strftime('%Y-%m-%d'))
    return a_year


def find_missing_dates(dates_to_check):
    complete_year = create_dates_list(time_period_days=2000)
    missing_dates = []
    for date in complete_year:
        if date not in dates_to_check:
            missing = date
            missing_dates.append(missing)
    return missing_dates


def get_first_day(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="max")

    first_day = hist.index[0]
    first_day = first_day.date()

    return first_day
