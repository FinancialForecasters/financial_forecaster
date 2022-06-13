# Financial Forecasters Capstone Project

## Table of Contents
- [Financial Forecasters Capstone Project](#Financial Forecasters Capstone Project)
  - [Table of Contents](#table-of-contents)
  - [Project Goal](#project-goal)
  - [Project Description](#project-description)
  - [How to Reproduce](#how-to-reproduce)
  - [Initial Questions](#initial-questions)
  - [Data Dictionary](#data-dictionary)
  - [Project Plan](#project-plan)
    - [1. Imports](#1-imports)
    - [2. Acquisition](#2-acquisition)
    - [3. Preparation](#3-preparation)
    - [4. Exploration](#4-exploration)
    - [5. Forecasting and Modeling](#5-forecasting-and-modeling)
    - [Deliverables](#deliverables)
    - [Final Report](#final-report)
  - [Key Findings](#key-findings)
  - [Recommendations](#recommendations)
  - [Next Steps](#next-steps)


## Project Goal

Our goal for the project was to predict the direction of Bitcoin's next day closing price using features related to supply and demand. These predictions were used as inputs to a trading strategy and profitability and risk were assessed.

## Project Description

For this project, daily price data for Bitcoin was acquired using Yahoo Finance. Several price transformations (technical indicators) were calculated based on the daily open, high, low, and close price of Bitcoin. Additional features related to the supply of Bitcoin, such as miner transactions and revenue data, were acquired as csvs from Blockchain.com. Twitter sentiment data was acquired from both a Kaggle dataset (for Tweets < 2019) and via scraping via the snscrape Python library. Exploratory data analysis was performed to investigate the relationship between these factors and returns. Based on the results of this analysis machine learning models were built with some combination of these features as inputs with the target being the direction of the next day's close. Finally, the model predictions were used as inputs to a simple trading strategy that decides when to buy or sell short Bitcoin, and the profitability and risk of this strategy assessed. 

## How to Reproduce 

1. Clone the repo (including the tidy.py and model.py modules as well as the csvs)
2. Libraries used:

- pandas
- matplotlib
- seaborn
- numpy
- scikit-learn
- statsmodels
- snscrape
- scipy
- <https://scipy.org/>
- TA-Lib
  - <https://mrjbq7.github.io/ta-lib/index.html>


## Initial Questions

1. Does high volatility result in above average returns?
1. Is social media sentiment predictive of Bitcoin returns?
1. 
1. 

## Data Dictionary

Variables |Definition
--- | ---
Index | Datetime in the format: YYYY-MM-DD. Time Zone: UTC
open | Price at open of the day
high | Highest price for day
low | Lowest price per day
close | Price at close of the day
volume | Amount in $USD traded for the day
fwd_log_ret | the log of tomorrow's close - log of today's close
fwd_close_positive | whether tomorrow's close is higher than today's
cross |
histy | 
month_9 | Encoded column for transaction during month 9 (September)
month_10 | Encoded column for transaction during month 10 (October)
day_20 | Encoded column for transaction on month day 20
day_1 | Encoded column for transaction on first day of month
day_9 | Encoded column for transaction on month day 9
atr_above_threshold_0.01 | True when today's ATR is above the historical (14 day) average ATR by the given threshold (0.01)
atr_above_threshold_0.05 | True when today's ATR is above the historical (14 day) average ATR by the given threshold (0.05)
atr_above_threshold_0.1 | True when today's ATR is above the historical (14 day) average ATR by the given threshold (0.1)
atr_above_threshold_0.2 | True when today's ATR is above the historical (14 day) average ATR by the given threshold (0.2)
atr_above_threshold_0.3 | True when today's ATR is above the historical (14 day) average ATR by the given threshold (0.3)
avg-fees-per-transaction | Amount in $USD of average fees per transaction (by day)
cost-per-transaction-percent | 
cost-per-transaction | 
difficulty |
hash-rate |
miners-revenue |
transaction-fees-to-miners |

## Project Plan

Method:

### 1. Imports

- Imports used can be found in `imports.py`. (Please ensure libraries are installed for package support).

### 2. Acquisition

- BTC trade data was acquired as a csv file from Yahoo Finance
- Miner features were acquired as csv files form Blockchain.com
- Tweets from Twitter were scraped from Twitter using the snscrape library

### 3. Preparation

- Preparation and cleaning consisted of:
  - renaming columns for readability.
  - changing data types where appropriate.
  - set the index to `datetime`.
  - for Tweets:
      - very short or blank Tweets were removed
      - VADER sentiment score was calculated for each Tweet
      - Sentiment scores were aggregated for each day using the mean value

### 4. Exploration

- I conducted an initial exploration of the data by examing relationships between each of the features and treated close price as a target.
- Next, I explored further using premier tools such as Pandas, Python, Statsmodels, etc..., to answer the initial questions posed above.
- Findings:
  - frequency analysis revealed potential price indicators.

### 5. Forecasting and Modeling

- I used data from 2022 April 26 from approximately 03:30 - 20:30 to determine if the candlestick close price, in conjuncture with the time index, could be used to determine future close prices, then modeled what the predicted values would like against the acutal values.

### Deliverables

### Final Report

## Key Findings

While one model alone was not effective at predicting future values, there may be a pattern of multiple models, that could at least recognize trade flags, if not predict them altogether.

## Recommendations

1. DO consider using the descriptive statistics to see highs and lows in the price of bitcoin over the past several hours and use that information, in conjunction with other sound trading principles, to find price points that are suitable for your portfolio.
2. DO NOT use the models in this project to make trade decisions. The predictions in this project are wildly inaccurate compared to the behavior of the actual bitcoin market.

## Next Steps

- explore a clustering model with the full set of candlestick features to glean an unsupervised machine's learning perspective.
- compare RMSE of Facebook's "Prophet" model to current models.


