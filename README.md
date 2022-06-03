# Binance Project

## Table of Contents
- [Project Goal](#project-goal)
- [Project Description](#project-description)
- [How to Reproduce](#how-to-reproduce)
- [Initial Questions](#initial-questions)
- [Data Dictionary](#data-dictionary)
- [Project Plan](#project-plan)
    - [Imports](#imports)
    - [Acquisition](#acquisition)
    - [Preparation](#preparation)
    - [Exploration](#exploration)
    - [Forecasting and Modeling](#forecasting-and-modeling)
    - [Deliverables](#deliverables)
    - [Final Report](#final-report)
- [Key Findings](#key-findings)
- [Recommendations](#recommendations)
- [Next Steps](#next-steps)


## Project Goal

This project analyzes recent bitcoin trade data in order to attempt price prediction.

Most of what I know about crypto analysis on Binance, I learned from Part Time Larry. Check out his deets:

- PartTimeLarry
- <https://hackingthemarkets.com/>

## Project Description

For this project, I acquired btcusd 1 minute kline data from theBinance API; <https://docs.binance.us/#introduction>. Tidying the data includes changing column labels, changing index to time index (`'close_time'` in this case) and finally the data is split into train, validate, test datasets. Data exploration delves into the descriptive statistics of the dataset. Further investigation includes up / down -sampling, frequency analysis, lag response, and autocorrelation. With a firm grasp of the data, I offer several models that attempt to predict the future price of btcusd trading pair. I used a last observed value (lov), average, 15 minute simple moving average from TAlib, and a basic Holt's linear trend. Root mean square errors (RMSE) are reported for comparison.

## How to Reproduce 

1. You will need an env.py file that contains the hostname, username and password for your Binance account. Please check the resources on their page for encrypted api access. Store that env file locally in the repository.
2. Clone my repo (including the tidy.py and model.py modules) (confirm .gitignore is hiding your env.py file)
3. Libraries used:

- pandas
- matplotlib
- seaborn
- numpy
- sklearn
- math
- statsmodels
- scipy
- <https://scipy.org/>
- TA-Lib
  - <https://mrjbq7.github.io/ta-lib/index.html>
- binance api
  - <https://www.binance.com/en/support/faq/360002502072>
  - <https://algotrading101.com/learn/binance-python-api-guide/>
  - <https://anaconda.org/conda-forge/python-binance>
- included in python-binance
  - websockets
    - <https://websockets.readthedocs.io/en/stable/>

## Initial Questions

1. What result will time-series-analysis have on previous binance data?
1. How accurate are predictions compared to actual values?
1. Can I predict the future price of bitcoin?

## Data Dictionary

Definitions for historical K-line data pulled from Binance API.
Variables |Definition
--- | ---
Index | Datetime in the format: YYYY-MM-DD
open | Price at open
high | Highest price for day
low | Lowest price per day
close | Price at close
volume | Amount in $USD traded for the day
fws_log_ret | 
fwd_close_positive |
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

Example data entry:

```text
1499040000000,      // Open time
"0.00386200",       // Open
"0.00386200",       // High
"0.00386200",       // Low
"0.00386200",       // Close
"0.47000000",  // Volume
1499644799999,      // Close time
"0.00181514",    // Quote asset volume
1,                // Number of trades
"0.47000000",    // Taker buy base asset volume
"0.00181514",      // Taker buy quote asset volume
"0" // Ignore.

```

## Project Plan

Method:

### 1. Imports

- Imports used can be found in `imports.py`. (Please ensure libraries are installed for package support).

### 2. Acquisition

- In this stage, I obtained btcusd trading pair data by querying the Binance REST API hosted at <https://api.binance.us/api/v3/klines>.

### 3. Preparation

- I cleaned and prepped the data by:
  - removing all observations that included null values.
  - renaming columns for readability.
  - changing data types where appropriate.
  - set the index to `datetime`.

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


