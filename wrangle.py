from imports import *
import tidy

def wrangle_df():
    '''
    Acquires initial dataframe from yfinance and then adds engineered features.
    '''
    # Our intial dataframe
    df = explore_df()
    # Adding macd engineered features
    df = macd_df(df)
    # Adding time engineered features
    df = time_features(df)
    # Adding atr feature
    df = add_ATR_feature(df)
    # Adding miner features
    df = add_miner_features(df)

    # Return df
    return df


