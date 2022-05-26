# acquire.py
'''
pull the latest 1000 candlestick entries from the binance api
'''

import sys
local_path = '/Users/hinzlehome/codeup-data-science/binance-project/'
sys.path.insert(0, local_path)
from utils.imports import *
# used for trouble shooting filepath issues
# import os
# print(os.getcwd())

def tidy_btcusd():
	if os.path.exists('/Users/hinzlehome/codeup-data-science/binance-project/csv/btcusd.csv'):
		print('cached csv')
		df = pd.read_csv('/Users/hinzlehome/codeup-data-science/binance-project/csv/btcusd.csv')
		return df
	else:
		payload = {'symbol':'BTCUSD','interval':'1m','limit':'1000'}
		r = requests.get('https://api.binance.us/api/v3/klines', params=payload)
		btcusd_json=r.json()
		btcusd_df=pd.DataFrame(btcusd_json)
		columns=['open_time','open','high','low','close','volume','close_time','quote_asset','number_of_trades','taker_buy_base_asset_vol','taker_buy_quote_asset_vol','ignore']
		btcusd_df.columns=columns
		btcusd_df.to_csv('/Users/hinzlehome/codeup-data-science/binance-project/csv/btcusd.csv', index=False)
		return btcusd_df

def model_btcusd(df):
	df.close_time=pd.to_datetime(df.close_time, unit='ms')
	df=df.set_index('close_time').sort_index()
	# about 17 hours of data
	train = df.loc[:'2022-04-25 15:31']
	# train is 12 hours
	validate =df.loc['2022-04-25 15:31':'2022-04-25 18:31'] 
	# validate is 3 hours
	test = df.loc['2022-04-25 18:31':]
	#test is ~2 hours
	return train, validate, test

def pre_cleaning(df):
	drops=['ignore']
	df=df.drop(labels=drops,axis=1)
	return df

def btcusd():
	df=tidy_btcusd()
	df=pre_cleaning(df)
	return model_btcusd(df)
