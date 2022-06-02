from imports import *
import tidy

def wrangle_df():
    '''
    Acquires initial dataframe from yfinance and then adds engineered features.
    '''
    # Our intial dataframe
    df = tidy.explore_df()
    # Adding macd engineered features
    df = tidy.macd_df(df)
    df = df.dropna()
    # Adding time engineered features
    df = tidy.time_features(df)
    # Adding atr feature
    df = tidy.add_ATR_feature(df)
    # Adding miner features
    df = tidy.add_miner_features(df)

    # Return df
    return df


