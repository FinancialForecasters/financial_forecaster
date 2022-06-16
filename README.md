# Financial Forecasters Capstone Project

## Table of Contents
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
In recent years cryptocurrencies have become better known and continue to inch closer to mainstream adoption as an investment option. This is evident with the most recent announcement by Fidelity that Bitcoin will be offered for 401ks in the near future. This type of commodity remains largely speculative and volatile, which creates ample space for profit if one can exploit it. With extensive industry promotion there is a constant increase in the number of small-scale retail traders entering the market, and this project aims to provide some guidance to them.   

For this project, daily price data for Bitcoin was acquired using Yahoo Finance. Several price transformations (technical indicators) were calculated based on the daily open, high, low, and close price of Bitcoin. Additional features related to the supply of Bitcoin, such as miner transactions and revenue data, were acquired as csvs from Blockchain.com. Twitter sentiment data was acquired from both a Kaggle dataset (for Tweets < 2019) and via scraping using the snscrape Python library. Exploratory data analysis was performed to investigate the relationship between these factors and returns. Based on the results of this analysis machine learning models were built with some combination of these features as inputs with the target being the direction of the next day's close. Finally, the model predictions were used as inputs to a simple trading strategy that decides when to buy or sell short Bitcoin, and the profitability and risk of this strategy assessed. 

## How to Reproduce 

1. Clone the repo (including the tidy, wrangle, explore, and model modules as well as the csvs)
2. Libraries used:

- pandas
- matplotlib
- seaborn
- numpy
- scikit-learn
- snscrape
- scipy
  - https://github.com/JustAnotherArchivist/snscrape
- TA-Lib (Technical Analysis Indicators)
  - <https://mrjbq7.github.io/ta-lib/index.html>


## Initial Questions

1. Does high volatility - quantified using ATR - result in above average returns?
1. Is social media sentiment predictive of Bitcoin returns?
1. Are any days of the week or month better for buying Bitcoin?
1. How does price momentum - quantified using MACD - affect returns?
1. Do factors related to mining, such as transaction revenue, affect returns?

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
cross | crossover indicator
histy | state of the MACD histogram
month_9 | Encoded column for transaction during month 9 (September)
month_10 | Encoded column for transaction during month 10 (October)
day_20 | Encoded column for transaction on month day 20
day_1 | Encoded column for transaction on first day of month
day_9 | Encoded column for transaction on month day 9
ATR | Technical analysis indicator used for measuring market volatility. 
MACF | Trend-following momentum indicator, shows relationship between two moving averages.
Volatility | Typically a measure of how fast a market is moving inside of a given range.
atr_above_threshold_0.01 | True when today's ATR is above the historical (14 day) average ATR by the given threshold (0.01)
atr_above_threshold_0.05 | True when today's ATR is above the historical (14 day) average ATR by the given threshold (0.05)
atr_above_threshold_0.1 | True when today's ATR is above the historical (14 day) average ATR by the given threshold (0.1)
atr_above_threshold_0.2 | True when today's ATR is above the historical (14 day) average ATR by the given threshold (0.2)
atr_above_threshold_0.3 | True when today's ATR is above the historical (14 day) average ATR by the given threshold (0.3)
avg-fees-per-transaction | Amount in $USD of average fees per transaction (by day)
cost-per-transaction-percent | miners revenue as percentage of the transaction volume
cost-per-transaction | miners revenue divided by the number of transactions
difficulty | A score of how difficult it is to produce a new block on the blockchain. 
hash-rate | Estimated number of terahashes per second the bitcoin network is performs over previous 24 hours
miners-revenue | Total value in USD of coinbase block rewards and transaction fees paid to miners
transaction-fees-to-miners | Value of all transaction fees paid to miners

## Project Plan

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

- We investigated the relationship between the features and the target, asking whether any features were drivers of returns. 

- Findings:
    - Few supply related features had correlation with returns. Miner Revenue per Transaction showed a weakly linear correlation with returns suggesting that as miner revenue per transaction increases we can expect returns to increase as well. This is contract to our initial hypothesis and indicates potential for a confounding variable.
    - Average next day returns are positive when current volatility measured by ATR is higher than historical volatility and in fact reach a peak when ATR is 20% greater than average. When volatility is greater than 20% above average returns decline and actually turn negative when volatility is very high.
    - Divergence between Twitter sentiment and Bitcoin's closing price may have helped a trader stay out of the market when prices were about to crash. 
    - Watching momentum via MACD can help us understand if a security is gaining or losing popularity in the market. Generally, positive momentum will drive the price higher while the opposite is also considered true. Positive momentum is indicated by a MACD value greater than the signal value.
    - Returns are highly variable depending on the day of the month and the month. Three specific days and two months show a statistically significant relationship with returns.

### 5. Modeling

- We took the results of our analysis of the various supply and demand factors such as miner revenue, Twitter sentiment and volatility, and used them as features for classification models. Our prediction target was the direction of the next day’s return - would tomorrow’s closing price be higher than today’s? As our ranking metric we used average percent return divided by standard deviation of percent returns, which is a form of risk to reward ratio. To  understand how the model would perform on unseen data the data was split as follows. The very last month of data (May 2022) was withheld for testing of the final model, while the remaining data (going back to 2014) was split into four rolling windows each approximately 3.5 years long. Within each of these windows the data was split into train and validate sets. Each classification model was trained on the training set and then used for predictions on the validate set. The average of the scores on the four validate sets was used to determine each model’s overall performance.

### Deliverables

1. Presentation June 16, 2022([Slides](https://docs.google.com/presentation/d/1GMxa0X7DhOl-887oxuFqR4z95HsYQMSZ7hmThKwWH_A/edit?usp=sharing))
2. This README and Repository with [Final Notebook](https://github.com/FinancialForecasters/financial_forecaster/blob/main/ff_final_notebook.ipynb) 
3. [White paper](https://docs.google.com/document/d/1fW-3TGXpA7L-P8lwt-DrrKFtTjyhpk_B3PtscvZsKYU/edit?usp=sharing)


## Key Findings

We experimented with classification and regression models to predict the relative direction of the next day’s closing price, but their performance severely deteriorated in the final test evaluations. The features we used show promise statistically but did not perform well in their current utilization beyond train and validate sampling. Any correlation or relationship is likely fleeting and would need to be exploited quickly before other market participants catch on. 

## Recommendations

1. Further research is needed to determine features predictive of the daily returns of Bitcoin.
2. Perform further engineering of the existing features we acquired and tune them specifically for short-term trading.
3. Try other modeling methods, such as shorter training periods and more sophisticated models
4. DO NOT use the models in this project to make trade decisions. The predictions in this project are wildly inaccurate compared to the behavior of the actual bitcoin market.

## Next Steps

1. Develop the features explored in this project further and test more sophisticated modeling techniques. 
