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
    # Adding time engineered features
    df = tidy.time_features(df)
    # Adding atr feature
    df = tidy.add_ATR_feature(df)
    # Adding miner features
    df = tidy.add_miner_features(df)
    # Adding Twitter sentiment data
    df = tidy.add_twitter_sentiment(df)
    # Adding volume feature
    df = tidy.add_obv_feature(df)
    # Drop nulls
    df = df.dropna()
    # Convert index of df to datetime
    df.index = pd.to_datetime(df.index)
    # Return df
    return df


def wrangle_3_df():
    '''
    
    '''
    # Our intial dataframe
    df = tidy.explore_df()
    # Adding macd engineered features
    df = tidy.macd_df(df)
    # Adding time engineered features
    df = tidy.time_features(df)
    # Adding atr feature
    df = tidy.add_ATR_feature(df)
    # Adding Twitter sentiment data
    df = tidy.add_twitter_sentiment(df)
    # Adding volume feature
    df = tidy.add_obv_feature(df)
    # Adding bitcoin circulation feature
    df = tidy.add_df_circ(df)
    # Adding miner features
    df = tidy.miner_features_3yr(df)
    # Drop nulls
    df = df.dropna()
    # Convert index of df to datetime
    df.index = pd.to_datetime(df.index)
    # Return df
    return df
