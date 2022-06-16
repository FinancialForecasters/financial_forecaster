'''
pull the latest 1000 candlestick entries from the binance api
'''

from numpy import histogram
from imports import *

def csv_btcusd():
    '''
    this function returns the yfinance.com, bitcoin trading data in a pandas dataframe.
    '''
    # check to see if the file is on root
    if os.path.exists('BTC-USD.csv'):
        # if so, print confirmation
        print('cached csv')
        # import csv file into pandas as 'df'
        df = pd.read_csv('BTC-USD.csv')
        # return df
        return df
    # if there's no file on root, this function returns nothing
    else:
        # payload = {'symbol':'BTCUSD','interval':'1m','limit':'1000'}
        # r = requests.get('https://api.binance.us/api/v3/klines', params=payload)
        # btcusd_json=r.json()
        # btcusd_df=pd.DataFrame(btcusd_json)
        # columns=['open_time','open','high','low','close','volume','close_time','quote_asset','number_of_trades','taker_buy_base_asset_vol','taker_buy_quote_asset_vol','ignore']
        # btcusd_df.columns=columns
        # btcusd_df.to_csv('/Users/hinzlehome/codeup-data-science/binance-project/csv/btcusd.csv', index=False)
        # return btcusd_df
        return None

def pre_cleaning(df):
    '''
    this function accepts a df of bitcoin data and begins preparation for the data science pipeline
    '''
    # add columns for removal to this list
    drops=['Adj Close']
    # drop columns
    df=df.drop(labels=drops,axis=1)
    # rename columns for sake of convenience
    df=df.rename(columns={'Date':'date','Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
    # set column to datetime
    df.date=pd.to_datetime(df.date)
    # set index as datetime
    df=df.set_index('date').sort_index()
    # df.date=df.date.strftime('%Y-%m-%d')
    # returns df ready for exploration
    return df

def add_targets(df):
    """ Adds target to dataframe. Returns dataframe with additional features """
    ###### TARGETS ######
    # forward 1 day log returns
    df["fwd_log_ret"] = np.log(df.close.shift(-1)) - np.log(df.close)
    # forward standard returns
    df["fwd_ret"] = df.close.shift(-1) - df.close
    # forward pct change
    df["fwd_pct_chg"] = df.close.pct_change(1).shift(-1)
    # binary positive vs negative next day return
    df["fwd_close_positive"] = df.fwd_ret>0
    
    # drop any remaining nulls
    df = df.dropna()
    
    return df

def add_ATR_feature(df):
    """Adds columns with boolean of whether current and historical ATR 
    is greater than the ATR threshold (percentage)"""
    
    df_calc = df.copy()

    # Calculate the 14 day ATR and add it as column to df
    df_calc['ATR_14'] = talib.ATR(df_calc.high, df_calc.low, df_calc.close, 14)
    # Calculate the rolling 14 day average of ATR and add it as column to df
    df_calc['avg_atr_14'] = df_calc.ATR_14.rolling(14).mean()
    # Calculate the percentage current 14 day ATR is above/below the rolling mean
    df_calc['atr_vs_historical'] = (df_calc.ATR_14 - df_calc.avg_atr_14)/df_calc.avg_atr_14
    
    thresholds_to_add = [0.01, 0.05, 0.1, 0.2, 0.3]
    
    for threshold in thresholds_to_add:
        df[f'atr_above_threshold_{threshold}'] = df_calc.atr_vs_historical>threshold
    
    return df

def finance_df():
    '''
    this function returns the yfinance.com, bitcoin trading data in a pandas dataframe.
    df is intended for modeling.
    '''
    # pull df from root
    df=csv_btcusd()
    # prepare df
    df=pre_cleaning(df)
    # add engineered features
    df=add_targets(df)
    # return df
    return model_btcusd(df)

def explore_df():
    '''
    this function returns the yfinance.com, bitcoin trading data in a pandas dataframe.
    df is intended for modeling.
    '''
    # pull df from root
    df=csv_btcusd()
    # prepare df
    df=pre_cleaning(df)
    # dumb datetime index always getting yeeted
    df.index = pd.to_datetime(df.index)
    return add_targets(df)


##### NLP PROCESSING #####
def basic_clean(article:str):
    """ Performs basic cleaning of text string, article, by switching all letters to lowecase, normalizing unicode characters, 
    and replacing everything that is not a letter, number, whitespace, or single quote."""
    # Convert text to lowercase
    article = article.lower()
    
    # Remove accented characteries. Normalize removes inconsistencies in unicode character encoding.
    # Encode converts string to ASCII and decode returns the bytes into string.
    article = unicodedata.normalize('NFKD', article)\
    .encode('ascii', 'ignore')\
    .decode('utf-8', 'ignore')

    # remove anything that is not a through z, a number, a single quote, or whitespace
    article = re.sub(r"[^a-z0-9'\s]", '', article)
    
    return article

def tokenize(article:str):
    """ Takes in a string, article, and tokenizes all words """
    
    tokenizer = nltk.tokenize.ToktokTokenizer()

    return tokenizer.tokenize(article, return_str=True)

def stem(article: str):
    """ Takes in a string, article, and returns text after applying stemming using Porter method """
    
    ps = nltk.porter.PorterStemmer()

    stems = [ps.stem(word) for word in article.split()]
    article_stemmed = ' '.join(stems)
    
    return article_stemmed

def lemmatize(article: str):
    """ Accepts string as argument, article, and returns text after applying lemmatization to each word """
    
    wnl = nltk.stem.WordNetLemmatizer()
        
    lemmas = [wnl.lemmatize(word) for word in article.split()]
    article_lemmatized = ' '.join(lemmas)

    return article_lemmatized

def remove_stopwords(article: str, extra_words: list, exclude_words: list):
    """ Accepts string (article) as argument and returns text after removing all the stopwords.
    extra_words: any additional stop words to include (these words will be removed from the article)
    exclude_words: any words we do not want to remove. These words are removed from the stopwords list and will remain in article """
    
    stopword_list = stopwords.words('english')

    [stopword_list.append(word_to_add) for word_to_add in extra_words if word_to_add not in stopword_list]
    [stopword_list.remove(to_remove) for to_remove in exclude_words if to_remove in stopword_list]

    words = article.split()
    filtered_words = [w for w in words if w not in stopword_list]

    # print('Removed {} stopwords'.format(len(words) - len(filtered_words)))

    article_without_stopwords = ' '.join(filtered_words)
    
    return article_without_stopwords

def prepare_df(df, column, extra_words = [], exclude_words = []):
    """Adds columns for cleaned, stemmed, and lemmatized data in dataframe. 
    Also adds in columns calculating the lengths and word counts. """
    # Create cleaned data column of content
    df['clean'] = df[column].apply(basic_clean).apply(tokenize).apply(remove_stopwords,
                                                       extra_words = extra_words,
                                                       exclude_words = exclude_words)
    
    # Create stemmed column with stemmed version of cleaned data
    df['stemmed'] = df.clean.apply(tokenize).apply(stem).apply(remove_stopwords,
                                                       extra_words = extra_words,
                                                       exclude_words = exclude_words)

    # Create lemmatized column with lemmatized version of cleaned data
    df['lemmatized'] = df.clean.apply(tokenize).apply(lemmatize).apply(remove_stopwords,
                                                       extra_words = extra_words,
                                                       exclude_words = exclude_words)
    
    # Calculates total length of readme based on number of characters
    df['original_length'] = df[column].str.len()
    df['stem_length'] = df.stemmed.str.len()
    df['lem_length'] = df.lemmatized.str.len()

    # Calculates total number of words (splitting up by whitespace)
    df['original_word_count'] = df[column].str.split().str.len()
    df['stemmed_word_count'] = df.stemmed.str.split().str.len()
    df['lemmatized_word_count'] = df.lemmatized.str.split().str.len()

    return df

## Miner features aka AJ Features

def add_csv(df, filename):
    '''
    This fuction will add a csv data to the main dataframe
    '''
    # read the CSV file and assign a variable
    filename_df = pd.read_csv(f'~/codeup-data-science/financial_forecaster/project_csvs/{filename}.csv')
    # change dtype of timestamp into pandas date
    filename_df.Timestamp = pd.to_datetime(filename_df.Timestamp).dt.date
    # reset index to datetime
    filename_df = filename_df.set_index('Timestamp').sort_index()
    # reset index to datetime for dataframe
    df.index = pd.to_datetime(df.index)
    # remove times to index
    df.index = df.index.date
    # add the CSV_dataframe to given dataframe
    df[filename] = filename_df
    # fill the nulls
    df.fillna(method='ffill', inplace=True)
    # retunrs a dataframe
    return df

def add_miner_features(df):  
    '''
    This functino will add all the miner CSVs to a main dataframe
    '''
    # add all the CSV files to a variable
    csv_filenames = ['avg-fees-per-transaction', 'cost-per-transaction-percent', 'cost-per-transaction', 'difficulty', 'hash-rate', 'miners-revenue', 'transaction-fees-to-miners']
    # loop each CSV into the dataframe using add_cvs function
    for filename in csv_filenames:
        add_csv(df, filename)
    # return df
    return df

## Miner Features for last 3 years
def add_csv_3yr(df, filename):
    '''
    This fuction will add a csv data to the main dataframe
    '''
    # read the CSV file and assign a variable
    print(f'reading {filename}')
    filename_df = pd.read_csv(f'~/codeup-data-science/financial_forecaster/project_csvs/{filename}.csv')
    # change dtype of timestamp into pandas date
    filename_df.Timestamp = pd.to_datetime(filename_df.Timestamp).dt.date
    # reset index to datetime
    filename_df = filename_df.set_index('Timestamp').sort_index()
    filename_df.columns = [filename]
    # reset index to datetime for dataframe
    df.index = pd.to_datetime(df.index)
    # remove times to index
    df.index = df.index.date
    # add the CSV_dataframe to given dataframe
    df = df.join(filename_df, how='right')
    df = df.dropna()
    # retunrs a dataframe
    return df

def miner_features_3yr(df):  
    '''
    This function will add all the miner CSVs to a main dataframe
    '''
    # add all the CSV files to a variable
    csv_filenames = ['avg-fees-per-transaction_3yr', 'cost-per-transaction-percent_3yr', 'cost-per-transaction_3yr', 'difficulty_3yr', 'hash-rate_3yr', 'miners-revenue_3yr', 'transaction-fees-to-miners_3yr']
    # loop each CSV into the dataframe using add_cvs function
    print('running loop')
    for filename in csv_filenames:
        df = add_csv_3yr(df, filename)
        df.index = pd.to_datetime(df.index)
        print(f'adding {filename}')
    # return df
    print('done')
    return df

def time_features(df):
    '''
    This function adds time features to the dataframe based on statistical significance with the target variable.
    '''
    alpha = .05
    overall_mean = df.fwd_log_ret.mean()
    # Obtaining stastistically significant month for increase or decrease in fwd_log_ret
    to_encode_month = []
    for m in df.index.month.unique():
        month_sample = df[df.index.month == m].fwd_log_ret
        t, p = stats.ttest_1samp(month_sample, overall_mean)
        if p/2 > alpha:
            continue
        else:
            to_encode_month.append(m)
    for m in to_encode_month:
        df['month_'+str(m)] = df.index.month == m
    
    # Obtaining statistically significant day of month for increase or decrease in fwd_log_ret
    to_encode_day = []
    for d in df.index.day.unique():
        day_sample = df[df.index.day == d].fwd_log_ret
        t, p = stats.ttest_1samp(day_sample, overall_mean)
        if p/2 > alpha:
            continue
        else:
            to_encode_day.append(d)
    for d in to_encode_day:
        df['day_'+str(d)] = df.index.day == d

    # Converting boolean features to ints
    for col in df.columns:
        if df[col].dtype == bool:
            df[col] = df[col].astype(int)

    # Return df
    return df

def macd_df(df):
    '''
    macd encoder
    '''
    # set up macd calculator from talib
    macd, signal, histo = talib.MACD(df.close,fastperiod=12, slowperiod=26, signalperiod=9)
    # add macd, signal, and histogram to original dataframe
    mac=pd.concat([macd,signal,histo],axis=1)
    # rename macd features
    mac=mac.rename(columns={0:'macd',1:'signal',2:'histo'})
    # remove nulls from the first 26 rows caused by macd
    mac=mac.drop(mac[mac.index<'2014-10-20'].index)
    # last sweep for nulls, set to zero if there are any left
    mac=mac.fillna(0)
    # the following code feature-izes the macd
    # create bools for postive histogram values
    cools=mac.histo>0
    # start point for feature classification
    start=cools[0]
    # empty list, will append feature values
    not_list=[]

    # this loop will add a 1 to the list 'not_list' if the histogram is positive, else 0
    for x in cools:
        if x:
            not_list.append(1)
        else:
            not_list.append(0)
    # take the histogram feature list, cast as a pandas series
    not_list=pd.Series(not_list, index=mac.index)
    # set bools for macd greater than signal
    bools=mac.macd>mac.signal
    # start point for feature classification
    yesterday=bools[0]
    # empty list, will append feature values
    list=[]

    # this loop will add a 1 to the list 'list' when the `macd` intersects the 'signal', else 0
    for today in bools:
        if today==yesterday:
            list.append(0)
            continue
        else:
            list.append(1)
            yesterday=today
    # take the macd crossover feature list, cast as a pandas series 
    list=pd.Series(list, index=mac.index)

    # crossover indicator appended to input df
    macker=pd.concat([df,mac,list,not_list],axis=1)
    # feature columns renamed
    macker=macker.rename({0:'cross',1:'histy'},axis=1)

    # return ... (mileage may vary with road conditions)
    return macker

def split_i(df):
    '''
    this function creates a train/ validate test set for time series analysis
    '''

    train = df.loc[:'2022-3-22']

    # train split is majority of dataset, 2014-09:2022-03
    train = df.loc[:'2022-3-22']
    # validate split is second to last month of the dataset

    validate =df.loc['2022-03-23':'2022-04-23'] 
    return train, validate

def split_ii(df):
    '''
    this function creates a train/ validate test set for time series analysis
    ''' 
    # train split absorbs validate split from `split_i`
    train = df.loc[:'2022-04-24']
    # test split is last month of entire dataset
    test = df.loc['2022-04-25':]
    return train, test

    

def add_df_circ(df):
    '''
    This function adds in circulation information and engineers stock-flow features on a day to day change.
    '''
    df_circ = pd.read_csv('circulation_btc.csv')
    df_circ['Timestamp'] = pd.to_datetime(df_circ['Timestamp'])
    df_circ = df_circ.set_index('Timestamp')
    df_circ = df_circ.resample('D').mean()
    df = df.join(df_circ)
    
    df['flow'] = df['total-bitcoins'] - df['total-bitcoins'].shift(1)
    # Add a new column that is the daily df.total-bitcoins to df.flow ratio
    df['stock_flow_ratio'] = df['flow'] / df['total-bitcoins']
    # Column for if stock_flow_ratio of today went up or down from yesterday
    df['stock_flow_ratio_change'] = df['stock_flow_ratio'].shift(1) - df['stock_flow_ratio']
    return df


def add_twitter_sentiment(df, filepath = './project_csvs/twitter_sentiment_btc.csv'):
    """ Adds the Twitter Sentiment (average per day) to the prices df """
    
    # read twitter data from csv
    twitter_sentiment = pd.read_csv(filepath, index_col = 0)
    
    # Converts index to datetime index
    twitter_sentiment.index = pd.to_datetime(twitter_sentiment.index)
    
    # Time not needed, pull out date for joined to prices/other features df
    twitter_sentiment.index = twitter_sentiment.index.date
    
    df = df.copy()
    
    df = pd.concat([df, twitter_sentiment], axis = 1)
    
    return df

def add_obv_feature(df, rolling_period = 30):
    """ Adds On Balance Volume derived feature, called obv_close_product to dataframe.
    This feature incorporates volume and price change to determine the cumulative positive and negative volume.
    
    Per Investopedia the feature can be used to confirm price movements:
    -If close price is near the rolling period high and OBV is near the 30 day high the obv_close_product will be close to 1, signaling an upward confirmation
    -If close price is near the rolling period low and OBV is near the 30 day low the obv_close_product will be close to 0, signaling a downward confirmation
    -If close price is near the rolling period high but OBV is near the 30 day low the obv_close_product will show a value between 0-1, implying indecision
    """
    
    obv_ind_df = df.copy()

    # Calculate On Balance Volume, which is the cumulative volume 
    obv_ind_df['obv'] = talib.OBV(df.close, df.volume)

    # Calculate the rolling 30 day max and min of OBV
    obv_ind_df['obv_period_max'] = obv_ind_df.obv.rolling(rolling_period, min_periods=0).max()
    obv_ind_df['obv_period_min'] = obv_ind_df.obv.rolling(rolling_period, min_periods=0).min()

    # Calculate the rolling 30 day max and min close
    obv_ind_df['close_period_max'] = obv_ind_df.close.rolling(rolling_period, min_periods=0).max()
    obv_ind_df['close_period_min'] = obv_ind_df.close.rolling(rolling_period, min_periods=0).min()

    # Create OBV convergence/divergence indicator 
    # Calculate the position of the close within the rolling n- day min and max of close
    obv_ind_df['close_within_n_period_channel'] = (obv_ind_df.close-obv_ind_df.close_period_min)/(obv_ind_df.close_period_max-obv_ind_df.close_period_min)

    # Calculate the position of the OBV within the rolling n-day min and max of obv
    obv_ind_df['obv_within_n_period_channel'] = (obv_ind_df.obv-obv_ind_df.obv_period_min)/(obv_ind_df.obv_period_max-obv_ind_df.obv_period_min)

    # Calculate product of close position within channel and obv position within channel
    # Values closer to 1 indicate confirmation of higher prices
    # Values close to 0 indicate confirmation of lower prices
    # Values between 0.1-0.9 would indicate indecision/mismatch between volume and price movement
    obv_ind_df['obv_close_product'] = (obv_ind_df.close_within_n_period_channel*obv_ind_df.obv_within_n_period_channel)
    
    df['obv_close_product'] = obv_ind_df.obv_close_product
    
    return df



def scrape_tweets(start_date= '2022-03-06', end_date = '2022-03-07', final_end_date = '2022-06-03', search_query = '#bitcoin', num_tweets_day = 100):
    """ Scrapes historical Tweets from Twitter using snscrape library. Returns dataframe of Tweets.
    start_date: date to start scraping from (YYYY-MM-DD)
    end_date: usually just the day after start_date (YYYY-MM-DD)
    final_end_date: last day to scrape
    search query: string representing search query (#....)
    num_tweets_day: number of tweets to scrape per day
    """
    
    # Creating list to append tweet data to
    tweets_list2 = []
    all_tweets = {}
    # Using TwitterSearchScraper to scrape data and append tweets to list
    # set initial start and end date for day by day scraping
    start_date = start_date
    end_date = end_date
    final_end_date = final_end_date
    
    # iterate through each day until reach final end date
    while start_date < final_end_date:
        print('scraping...',start_date, end_date)
        tweets_list2 = []
        for i,tweet in enumerate(sntwitter.TwitterSearchScraper(f'{search_query} since:{start_date} until:{end_date}').get_items()):
            print(i, end = "\r")
            if i>num_tweets_day:
                break
            tweets_list2.append([tweet.date, tweet.id, tweet.content, tweet.user.username])

        # Add day's tweets to dictionary    
        all_tweets[start_date] = (tweets_list2)
        
        # increment and change string representation of date
        start_date = datetime.strptime(start_date, '%Y-%m-%d') + timedelta(days=1)
        end_date = start_date + timedelta(days=1)
        start_date = datetime.strftime(start_date,'%Y-%m-%d')
        end_date = datetime.strftime(end_date,'%Y-%m-%d')
        print(start_date, final_end_date)

    # Creating a dataframe from the tweets list above
    tweets_df2 = pd.DataFrame(tweets_list2, columns=['Datetime', 'Tweet Id', 'Text', 'Username'])
    tweets_df2.index = pd.to_datetime(tweets_df2.Datetime)
    tweets_df2 = tweets_df2.sort_index()
    
    return all_tweets

def move_tweets_to_csv(all_tweets, filepath = './csv/tweets.csv'):
    """Saves all_tweets to a csv"""
    temp_df = pd.DataFrame()
    for key in all_tweets.keys():
        that_day = pd.DataFrame(all_tweets[key], columns = ['time','user','text','username'])
        temp_df = pd.concat([temp_df, that_day])
        
    temp_df.to_csv(filepath)
    print(f"Tweets saved to csv at {filepath}")

def macd_plot(df):
	plt.rc('font',size=20)
	fig, ax1 = plt.subplots(figsize=(16, 9))

	rect = fig.patch
	rect.set_facecolor('white')
	plt.rc('font',size=20)

	color = 'tab:blue'
	# ax1.set_xlabel('Date',size=16)
	ax1.set_ylabel('"MACD" & "Signal" line', color = color,size=16)
	ax1.yaxis.set_major_formatter('${x:1.2f}')

	ax1.tick_params(axis ='y', labelcolor = color)

	df.macd.tail(150).plot()
	df.signal.tail(150).plot()
	plt.stem(df.index[-150:],df['histo'].tail(150),linefmt='pink',markerfmt='w,',label='hist')

	ax2 = ax1.twinx()
	
	color = 'black'
	ax2.set_ylabel('BTC Daily Close Price', color = color,rotation=270, labelpad=30,size=16)
	ax2.yaxis.set_major_formatter('${x:1.2f}')

	ax2.tick_params(axis ='y', labelcolor = color)

	_=df[(df.cross==1)&(df.histy==1)].tail(500)
	plt.scatter(y=_.close,x=_.index,c='green')

	_=df[(df.cross==1)&(df.histy==0)].tail(500)
	plt.scatter(y=_.close,x=_.index,c='red')

	df.close.tail(150).plot(color='black',alpha=.5)

	ax1.legend(loc=2)
	ax2.legend(loc=1)

	plt.tight_layout()

	plt.show()
