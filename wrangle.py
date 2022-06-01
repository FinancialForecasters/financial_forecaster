from imports import *

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


