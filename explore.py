from imports import *
from wrangle import *
import matplotlib.ticker as mtick
from matplotlib import gridspec

def plot_close_price_and_returns(price_df):
    """Plots chart of closing price with percentage returns below. Used for final presentation and report """
    
    # Get price data
    df = price_df.copy()
    
    fig = plt.figure(figsize = (16,8))
    gs = gridspec.GridSpec(2, 1, height_ratios = [2,1])
    ax0 = plt.subplot(gs[0])
    ax2 = plt.subplot(gs[0])
    # ax2 = ax0.twinx()
    ax0.plot(df.close)
    # ax2.plot(spy['Adj Close'], 'g', alpha = 0.2)
    ax0.set_ylabel('Close Price ($)', fontsize = 20)


    plt.setp(ax0.get_xticklabels(), visible = False)

    ax1 = plt.subplot(gs[1], sharex = ax0)
    ax1.plot(df.fwd_pct_chg*100)
    ax1.set_ylabel('Percent Returns', fontsize = 18)
    ax1.axhline(y=10, alpha = 0.3)
    ax1.axhline(y=-10, alpha = 0.3)
    ax1.yaxis.set_major_formatter(mtick.PercentFormatter())
    ax0.set_title('Bitcoin Close Price and Percentage Returns', fontsize = 20)
    ax0.set_xlim(left = pd.Timestamp('2018-01-01'), right = pd.Timestamp('2022-05-30'))

    plt.subplots_adjust(hspace = 0)

def perform_target_return_analysis(df, target = 'fwd_log_ret'):
    """Plots a histogram of returns"""
    
    if target == 'fwd_close_positive':
        df.fwd_close_positive.value_counts(normalize=True).plot.bar()
        plt.title('Classification Target Distribution - Next Day Close is Positive')
    else:
        average_log_return = df.fwd_log_ret.mean()
        df[target].hist(grid = False)
        plt.title(f'Daily Return Distribution\nAverage Return {average_log_return:.4f}')
        plt.xlabel('Log Return')


def perform_ATR_analysis(df, atr_threshold  = 0.05, make_plots = True, alpha = 0.05, print_results = False):
    """Performs analysis of returns based on current vs historical ATR.
    
    If the difference in current and historical ATR is greater than the ATR threshold (percentage)"""

    # Calculate the 14 day ATR and add it as column to df
    df['ATR_14'] = talib.ATR(df.high, df.low, df.close, 14)
    # Calculate the rolling 14 day average of ATR and add it as column to df
    df['avg_atr_14'] = df.ATR_14.rolling(14).mean()
    # Calculate the percentage current 14 day ATR is above/below the rolling mean
    df['atr_vs_historical'] = (df.ATR_14 - df.avg_atr_14)/df.avg_atr_14

    if make_plots:
    # Plot ATR 14 and close price
        fig, ax = plt.subplots(2,1,sharex=True)
        df.close.plot(ax = ax[0])
        ax[0].set_title('Close price of BTC')
        df.avg_atr_14.plot(ax = ax[1])
        ax[1].set_title('Rolling 14 day average of 14 day ATR')

    df['atr_above_threshold'] = df.atr_vs_historical>atr_threshold

    
    avg_return_above_threshold = round(df[df.atr_above_threshold].fwd_log_ret.mean(),6)

    # Perform one sample t-test -> is the average return of the high volatility days significantly greater than overall mean?
    if print_results:
        print(f"Percentage of observations above threshold: {df.atr_above_threshold.mean():.2%}")
        print(f"Average next day return when ATR above threshold: {avg_return_above_threshold}")
        print(f"which is: {round(df[df.atr_above_threshold].fwd_log_ret.mean()/df.fwd_log_ret.mean(),3)} times overall average")
        
        t,p = stats.ttest_1samp(df[df.atr_above_threshold].fwd_log_ret, df.fwd_log_ret.mean())

        if ((t>0)&(p/2<alpha)):
            print("Results significant!: t is >0",t>0,"p/2 < alpha",p<alpha)
        else:
            print("Fail to reject null hypothesis")
        
    return avg_return_above_threshold

def plot_consolidated_atr_analysis(df):
    """ Plot barchart showing average returns by percentage current ATR above historical """
    
    thresholds_to_try = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    returns = {}
    for threshold in thresholds_to_try:
        avg_return = perform_ATR_analysis(df, atr_threshold=threshold, make_plots=False)
        returns[threshold] = avg_return
        
    pd.DataFrame(returns, index = ['avg_return']).T.plot.bar(legend=False)
    plt.title('Average Next Day Return when ATR is \nAbove 14 Day Average ATR by Threshold Amount')
    plt.xlabel('Threshold')
    plt.ylabel('Average Return')
    plt.axhline(df.fwd_log_ret.mean(), color = 'r')
    plt.annotate('Overall Mean', (5, 0.0017))
    
def plot_sentiment_and_close(price_df, csv_file = './project_csvs/weekly_consolidated_later_sentiment.csv'):
    """Plot aggregated Twitter sentiment with close price going back to 2019 """
    
    # Read in aggregated tweet csv 
    lt = pd.read_csv(csv_file, index_col = 'time', parse_dates=True)
    
    # Read in price data
    df = price_df.copy()

    fig, ax = plt.subplots(figsize=(18,8))
    
    # Plot sentiment
    sentiment = ax.plot(lt, color = 'salmon', label = 'sentiment')
    ax.set_xlabel("Date", fontsize = 24)
    ax.set_ylabel("Average Twitter Sentiment", fontsize = 24)
    ax.tick_params(axis = 'both', which = 'major', labelsize = 18)

    # Plot close price data
    ax2 = ax.twinx()
    close_price = ax2.plot(df.loc[df.index>'2019-09-19'].close, color = 'mediumseagreen', label = 'close price')
    ax2.set_ylabel("Close Price ($)", fontsize = 24)
    
    # Clean up the legend
    lns = sentiment + close_price
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc = 2, fontsize = 24)
    
    plt.yticks(fontsize=18)
    plt.tight_layout()
    plt.title('Twitter Sentiment and Bitcoin Close Price', fontsize = 24)
    plt.show()

