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
	train = df.loc[:'2017']
	# train is 12 hours
	validate =df.loc['2018':'2021'] 
	# validate is 3 hours
	test = df.loc['2021':]
	#test is ~2 hours
	return train, validate, test

def pre_cleaning(df):
	drops=['Adj Close']
	df=df.drop(labels=drops,axis=1)
	df=df.rename(columns={'Date':'date','Open':'open','High':'high','Low':'low','Close':'close','Volume':'volume'})
	df.date=pd.to_datetime(df.date, utc=True)
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

def finance_df():
	df=csv_btcusd()
	df=pre_cleaning(df)
	df=add_targets(df)
	return model_btcusd(df)

def explore_df():
	df=csv_btcusd()
	df=pre_cleaning(df)
	return add_targets(df)
