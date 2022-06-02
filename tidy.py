# acquire.py
'''
pull the latest 1000 candlestick entries from the binance api
'''

from imports import *

def csv_btcusd():
	if os.path.exists('BTC-USD.csv'):
		print('cached csv')
		df = pd.read_csv('BTC-USD.csv')
		return df
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

def model_btcusd(df):
	# about 17 hours of data
	train = df.loc[:'2022-3-22']
	# train is 12 hours
	validate =df.loc['2022-03-23':'2022-04-24'] 
	# validate is 3 hours
	test = df.loc['2022-04-25':]
	#test is ~2 hours
	return train, validate, test

def pre_cleaning(df):
	drops=['Adj Close']
	df=df.drop(labels=drops,axis=1)
	df=df.rename(columns={'Date':'date','Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
	df.date=pd.to_datetime(df.date, utc=True)
	# df.date=df.date.strftime('%Y-%m-%d')
	df=df.set_index('date').sort_index()
	return df

def btcusd():
	df=tidy_btcusd()
	df=pre_cleaning(df)
	return model_btcusd(df)

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
	df=csv_btcusd()
	df=pre_cleaning(df)
	df=add_targets(df)
	return model_btcusd(df)

def explore_df():
	df=csv_btcusd()
	df=pre_cleaning(df)
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

    macd, signal, histo = talib.MACD(df.close,fastperiod=12, slowperiod=26, signalperiod=9)
    mac=pd.concat([df,macd,signal,histo],axis=1)
    mac=mac.rename(columns={0:'macd',1:'signal',2:'histo'})
    mac=mac.drop(mac[mac.index<'2014-10-20'].index)
    mac=mac.fillna(0)
    cools=mac.histo>0
    start=cools[0]
    not_list=[]

    for x in cools:
        if x:
            not_list.append(1)
        else:
            not_list.append(0)

    not_list=pd.Series(not_list, index=mac.index)
    bools=mac.macd>mac.signal
    yesterday=bools[0]
    list=[]

    for today in bools:
        if today==yesterday:
            list.append(0)
            continue
        else:
            list.append(1)
            yesterday=today

    list=pd.Series(list, index=mac.index)

    # crossover indicator
    macker=pd.concat([df,list,not_list],axis=1)
    macker=macker.rename({0:'cross',1:'histy'},axis=1)

    return macker
    