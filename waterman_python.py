from imports import *
import tidy

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


df = tidy.csv_btcusd()
df = tidy.pre_cleaning(df)
df = tidy.add_targets(df)

df = time_features(df)

df_circ = pd.read_csv('circulation_btc.csv')

# df_circ.Timestamp to datetime object
df_circ['Timestamp'] = pd.to_datetime(df_circ['Timestamp'], utc=True)
# df_circ.Timestamp as index
df_circ = df_circ.set_index('Timestamp')
# Mean index by day
df_circ = df_circ.resample('D').mean()
# Add to df
df = df.join(df_circ)
# Drop NaN
df = df.dropna()
# Add a new column of change in total_bitcoin between today and yesterday
df['flow'] = df['total-bitcoins'] - df['total-bitcoins'].shift(1)
# Add a new column that is the daily df.total-bitcoins to df.flow ratio
df['stock_flow_ratio'] = df['flow'] / df['total-bitcoins']
# Column for if stock_flow_ratio of today went up or down from yesterday
df['stock_flow_ratio_change'] = df['stock_flow_ratio'].shift(1) - df['stock_flow_ratio']




