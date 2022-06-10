from imports import *
from sklearn.metrics import mean_squared_error, classification_report, accuracy_score, r2_score
import matplotlib.dates as mdates
from matplotlib.dates import DateFormatter

def baseline_selection(y_train, target):
    '''
    This function takes our train and validate y components and computes a mean and median baseline for modeling purposes. It compares their
    RMSE values and returns whichever is a better usecase. 
    '''
    # Creation of mean value and adding to our dataframes
    pred_mean = y_train[target].mean()
    y_train['pred_mean'] = pred_mean
    # Creation of median value and adding to our dataframes
    pred_median = y_train[target].median()
    y_train['pred_median'] = pred_median
    # Evaluating RMSE value for mean baseline
    rmse_train_mean = mean_squared_error(y_train[target], y_train.pred_mean)**(1/2)
    # Evaluating RMSE value for median baseline
    rmse_train_med = mean_squared_error(y_train[target], y_train.pred_median)**(1/2)

    # Determine which is better to use as baseline
    if rmse_train_mean < rmse_train_med:
        print(f'Mean provides a better baseline. Returning mean RMSE of {rmse_train_mean}.')
        return rmse_train_mean
    else: 
        print(f'Median provides a better baseline. Returning median RMSE of {rmse_train_med}.')
        return rmse_train_med

def ols_model(x_train, y_train, x_validate, y_validate, target):
    '''
    This function takes in our x and y components for train and validate and prints the results of RMSE for train and validate modeling.
    '''
    # Creation and fitting of the model
    lm = LinearRegression(normalize=True)
    lm.fit(x_train, y_train[target])
    # Prediction creation and RMSE value for train
    y_train['ret_lm_pred'] = lm.predict(x_train)
    rmse_train = round(mean_squared_error(y_train[target], y_train.ret_lm_pred)**(1/2), 6)
    # Prediction creation and RMSE for validate
    y_validate['ret_lm_pred'] = lm.predict(x_validate)
    rmse_validate = round(mean_squared_error(y_validate[target], y_validate.ret_lm_pred)**(1/2), 6)
    # Returning the RMSE values
    return rmse_train, rmse_validate


def rolling_split_train_and_test(df, features_to_use, features_to_scale, target, other_targets, num_splits = 5, test_size = 30):
    """ Splits dataset into train and test sets. Returns list of X_trains, X_tests, y_trains, y_tests
    
    Arguments:
    df: the full dataset included features and target(s)
    features_to_use: list of features to be used in the model
    features_to_scale: list of features to scale
    target: the desired target
    num_splits: number of datasets to create
    test_size: number of datapoints to use for test sets
    """
    
    other_targets.append(target)
    # Create the initial X and y matrices
    # X consists of only selected features
    # Withholding the final <<test_size>> to save for a final test on unseen data
    X = df[features_to_scale].iloc[:-test_size]
    y = df[other_targets].iloc[:-test_size]
    
    # Instantiate Time Series Split
    tscv = TimeSeriesSplit(n_splits = num_splits, test_size = test_size)

    # Initialize the muliplier variable, which is used to shift the start date of train set forward
    multiplier = 0

    train_X_sets = []
    test_X_sets = []
    train_y_sets = []
    test_y_sets = []

    # Iterate through the splits created by tscv and create list of X and y train and test sets
    for train_index, test_index in tscv.split(X):
        # Sets up the rolling part of the split by moving the start index of forward by test_size for each split s
        # First split starts at 0, second starts at index = test_size, s split starts at 0+(s-1)*(test_size)
        train_index = train_index[multiplier * test_size:]

        # print("Train:", df.index[train_index], "Test:", df.index[test_index])
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        # Perform scaling after the split so scaler can be fit to each train set individually
        X_train_scaled, X_test_scaled = scale_X(X_train, X_test, features_to_use, features_to_scale, scaler_type = StandardScaler())
        # Increments multiplier for next split. The product of the multiplier and test size determine the starting index for the next train set
        multiplier += 1

        train_X_sets.append(X_train_scaled)
        test_X_sets.append(X_test_scaled)
        train_y_sets.append(y_train)
        test_y_sets.append(y_test)

    return train_X_sets, test_X_sets, train_y_sets, test_y_sets

def scale_X(X_train, X_test, features_to_use, features_to_scale, scaler_type):
    """ Scales X_train, X_test using scaler_type. Only scales features in features_to_scale. """
    scaler = scaler_type
    
    X_train_scaled = pd.concat([X_train.drop(columns = features_to_scale),
                                pd.DataFrame(data = scaler.fit_transform(X_train[features_to_scale]), 
                                             columns = features_to_scale, index = X_train.index)], 
                               axis=1)
    X_test_scaled = pd.concat([X_test.drop(columns = features_to_scale),
                                pd.DataFrame(data = scaler.transform(X_test[features_to_scale]), 
                                             columns = features_to_scale, index = X_test.index)], 
                               axis=1)
    
    return X_train_scaled, X_test_scaled

def train_classifier(X_train, y_train, target = 'fwd_close_positive', class_model = DecisionTreeClassifier(max_depth = 5, min_samples_leaf=2)):
    """ Trains classifer for inputted model. 
    Arguments:
    X_train: features (scaled data)
    y_train: target
    target: column name for target
    class_model: the model to use
    
    Outputs dictionary of classification report, the classifier object, and numpy array of predictions """
    
    # Extracts name of the algorithm
    algorithm_name = class_model.__repr__().split('()')[0].split('(')[0]
    
    # Extracts the hyperparameters
    model_hyperparameters = class_model.get_params()
    
    # Fits classifier to train set
    clf = class_model.fit(X_train, y_train[target])
    
    # Predict on train
    train_predictions = class_model.predict(X_train)
    
    # Generate classification report for train
    train_classification_report = classification_report(y_train[target], train_predictions, output_dict=True)
    
    # Add additional data for function output
    train_classification_report['algorithm'] = algorithm_name
    
    train_classification_report['hyperparameters'] = model_hyperparameters
    
    train_classification_report['features_used'] = list(X_train.columns)
    
    return train_classification_report, clf, train_predictions

def test_classifier(X_test, y_test, clf, target='fwd_close_positive'):
    """ Tests the classifier on test set """
    
    # Extracts name of the algorithm
    algorithm_name = clf.__repr__().split('()')[0].split('(')[0]
    
    # Extracts the hyperparameters for model
    model_hyperparameters = clf.get_params()
    
    # Predict on test
    test_predictions = clf.predict(X_test)
        
    # Generate classification report for test
    test_classification_report = classification_report(y_test[target], test_predictions, output_dict=True)
    
    # Add additional data for function output
    test_classification_report['algorithm'] = algorithm_name
    
    test_classification_report['hyperparameters'] = model_hyperparameters
    
    test_classification_report['features_used'] = list(X_test.columns)
    
    return test_classification_report, test_predictions
def compute_class_baseline_accuracy(y, target):
    """ Determine baseline from train and calculate accuracy on train """
    
    # use the mode of y (most common category) as baseline
    baseline_selection = y[target].mode()[0]
    
    # return accuracy
    return baseline_selection, (y[target] == baseline_selection).mean()

def compute_baseline_accuracy_on_test(y, baseline_selection, target):
    """ Use baseline selection from train to calculate performance on test/validate """
    
    return (y[target] == baseline_selection).mean()

def test_class_baseline(y_train, y_test, target, set_number, print_results = True):
    """ Calculates baseline performance for classification by using the mode of the value in train """
    # Calculate baseline performance
    baseline_selection, baseline_train_accuracy = compute_class_baseline_accuracy(y_train, target)
    baseline_test_accuracy = compute_baseline_accuracy_on_test(y_test, baseline_selection, target)
    
    if print_results:
        print("Split: ", set_number)
        print(f"Train: {y_train.index.min().date()} - {y_train.index.max().date()}")
        print(f"Test: {y_test.index.min().date()} - {y_test.index.max().date()}")
        print("Baseline selection from train: ", baseline_selection)
        print("Baseline train accuracy: ", round(baseline_train_accuracy,2))
        print("Baseline performance on validate: ", round(test_accuracy,2))
    
    return baseline_selection, baseline_train_accuracy, baseline_test_accuracy

def train_and_test_dataset(model_under_test, X_train, X_test, y_train, y_test, target, print_results = True):
    """Test one model by fitting on train and testing on test set"""
    
    # print(model_under_test.__repr__().split('()')[0])
    # Train classifier
    train_classification_report, clf, train_predictions = train_classifier(X_train, y_train, target = target, class_model = model_under_test)
    
    # Test classifier on test set
    test_classification_report, test_predictions = test_classifier(X_test, y_test, clf, target = target)
    
    if print_results:
        print("Train accuracy: ", train_classification_report['accuracy'])  
        print("Test accuracy: ", test_classification_report['accuracy'])
    
    return train_classification_report, test_classification_report, train_predictions, test_predictions

def consolidate_class_model_results(dataset_number, all_classification_model_results, train_classification_report,
                                    test_classification_report, train_equity_results, test_equity_results):
    """ Consolidates modeling results from each dataset split into one dataframe """
    
    # Compute dropoff from train to test
    train_to_test_dropoff = train_classification_report['accuracy']-test_classification_report['accuracy']
    
    # Compute percent dropoff
    train_to_test_dropoff_percent = train_to_test_dropoff/train_classification_report['accuracy']
    
    # Compute average return
    avg_train_return = train_equity_results['return_achieved'].mean()
    avg_test_return = test_equity_results['return_achieved'].mean()

    # Compute average percent return 
    avg_pct_train_return = train_equity_results['pct_return_achieved'].mean()
    avg_pct_test_return = test_equity_results['pct_return_achieved'].mean()

    # Compute standard deviation of returns
    std_train_return = train_equity_results['return_achieved'].std()
    std_test_return = test_equity_results['return_achieved'].std()
    
    # Compute standard deviation of percent returns
    std_pct_train_return = train_equity_results['pct_return_achieved'].std()
    std_pct_test_return = test_equity_results['pct_return_achieved'].std()
    
    # Compute Quasi - Sharpe Ratio (avg pct return / standard deviation of pct returns)
    if std_train_return != 0:
        pct_return_to_std_train = avg_pct_train_return/std_pct_train_return
    else:
        pct_return_to_std_train = 0
        
    if std_test_return != 0:
        pct_return_to_std_test = avg_pct_test_return/std_pct_test_return
    else:
        pct_return_to_std_test = 0
    
    
    # hyperparameters and features used are inputted as strings - would prefer list but need to investigate how
    return pd.concat([all_classification_model_results, 
                      pd.DataFrame({'dataset_number':dataset_number,
                          'algorithm': train_classification_report['algorithm'],
                         'hyperparameters': str(train_classification_report['hyperparameters']),
                         'features_used': str(list(train_classification_report['features_used'])),
                         'train_accuracy':train_classification_report['accuracy'],
                         'test_accuracy':test_classification_report['accuracy'],
                         'train_to_test_dropoff_pct': train_to_test_dropoff_percent,
                         'avg_train_return':avg_train_return,
                         'avg_test_return':avg_test_return,
                         'pct_avg_train_return':avg_pct_train_return,
                         'pct_avg_test_return':avg_pct_test_return,
                         'pct_return_to_std_train':pct_return_to_std_train,
                         'pct_return_to_std_test':pct_return_to_std_test},
                       index = [None])],ignore_index = True).sort_values(by=['train_to_test_dropoff_pct',
                                                                                        'pct_return_to_std_test'],
                                                                                    ascending=[True, False])
    

def consolidate_baseline_results(all_baseline_results, dataset_number,baseline_train_accuracy, baseline_test_accuracy,train_baseline_equity_results, test_baseline_equity_results):
    """ Consolidates the baseline results into one dataframe """
    
    # Calculates the train to test dropoff in accuracy
    train_to_test_dropoff = baseline_train_accuracy - baseline_test_accuracy
    
    # Calculate the percentage dropoff in train to test accuracy
    train_to_test_dropoff_percent = train_to_test_dropoff/baseline_train_accuracy
    
    
    # Compute average return
    avg_train_return = train_baseline_equity_results['return_achieved'].mean()
    avg_test_return = test_baseline_equity_results['return_achieved'].mean()

    # Compute average percent return 
    avg_pct_train_return = train_baseline_equity_results['pct_return_achieved'].mean()
    avg_pct_test_return = test_baseline_equity_results['pct_return_achieved'].mean()

    # Compute standard deviation of returns
    std_train_return = train_baseline_equity_results['return_achieved'].std()
    std_test_return = test_baseline_equity_results['return_achieved'].std()
    
    # Compute standard deviation of percent returns
    std_pct_train_return = train_baseline_equity_results['pct_return_achieved'].std()
    std_pct_test_return = test_baseline_equity_results['pct_return_achieved'].std()
    
    # Compute Quasi - Sharpe Ratio (avg pct return / standard deviation of pct returns)
    if std_pct_train_return != 0:
        pct_return_to_std_train = avg_pct_train_return/std_pct_train_return
    else:
        pct_return_to_std_train = 0
        
    if std_pct_test_return != 0:
        pct_return_to_std_test = avg_pct_test_return/std_pct_test_return
    else:
        pct_return_to_std_test = 0
    
    return pd.concat([all_baseline_results,
                      pd.DataFrame({'dataset_number':dataset_number,
                                    'train_accuracy':baseline_train_accuracy,
                                    'test_accuracy': baseline_test_accuracy,
                                    'train_to_test_dropoff_pct': train_to_test_dropoff_percent,
                                    'avg_train_return':avg_train_return,
                                    'avg_test_return':avg_test_return,
                                    'pct_avg_train_return':avg_pct_train_return,
                                    'pct_avg_test_return':avg_pct_test_return,
                                    'pct_return_to_std_train':pct_return_to_std_train,
                                    'pct_return_to_std_test':pct_return_to_std_test},
                                   index = [None])], ignore_index = True).drop_duplicates()
                                   
    
def consolidate_datasets(all_baseline_results, all_classification_dataset_model_results):
    """ Consolidate cross validation results from baseline and each model by taking the average of results from each dataset"""
    
    # Creates dataframe with average values across all splits of train accuracy, test accuracy, and dropoff
    consolidated_model_results =  (
        all_classification_dataset_model_results
        .groupby(
            ['hyperparameters','algorithm','features_used']
        )['train_accuracy', 'test_accuracy',
          'train_to_test_dropoff_pct', 'avg_train_return', 
          'avg_test_return', 'pct_return_to_std_train',
       'pct_return_to_std_test', 'pct_avg_train_return',
         'pct_avg_test_return']
        .mean()).reset_index()
    
    # Creates dataframe with average baseline results across all splits and drops dataset number
    baseline_results = pd.DataFrame(all_baseline_results.drop('dataset_number', 
                                                              axis = 1).mean()).T.assign(algorithm = 'baseline')
    
    return pd.concat([consolidated_model_results, 
                      baseline_results]).sort_values(by=['pct_return_to_std_test',
                                                         'train_to_test_dropoff_pct',
                                                         ],ascending=[False, True])

def get_equity_results(y_train, y_test, train_predictions, test_predictions):
    """ Generate equity curve based on model predictions """
    
    train_equity_results = pd.DataFrame(y_train)
    test_equity_results = pd.DataFrame(y_test)
    
    train_equity_results['model_prediction'] = train_predictions
    test_equity_results['model_prediction'] = test_predictions
    
    # Determines how the strategy should trade when model prediction is negative close
    train_equity_results.model_prediction = np.where(train_equity_results.model_prediction==0, -1,1)
    test_equity_results.model_prediction = np.where(test_equity_results.model_prediction==0, -1,1)
    
    train_equity_results['return_achieved'] = train_equity_results['model_prediction']*train_equity_results['fwd_ret']
    test_equity_results['return_achieved'] = test_equity_results['model_prediction']*test_equity_results['fwd_ret']
    
    train_equity_results['pct_return_achieved'] = train_equity_results['return_achieved']/y_train.close
    test_equity_results['pct_return_achieved'] = test_equity_results['return_achieved']/y_test.close

    return train_equity_results, test_equity_results

def get_class_baseline_equity_results(y_train, y_test, baseline_selection):
    """ Gets equity results for the baseline (classification) """
    
    train_baseline_equity_results = pd.DataFrame(y_train)
    test_baseline_equity_results = pd.DataFrame(y_test)
    
    # Set the prediction equal to the baseline selection for both train and test
    train_baseline_equity_results['model_prediction'] = baseline_selection
    test_baseline_equity_results['model_prediction'] = baseline_selection
    
    train_baseline_equity_results['return_achieved'] = train_baseline_equity_results['model_prediction']*train_baseline_equity_results['fwd_ret']
    test_baseline_equity_results['return_achieved'] = test_baseline_equity_results['model_prediction']*test_baseline_equity_results['fwd_ret']
    
    train_baseline_equity_results['pct_return_achieved'] = train_baseline_equity_results['return_achieved']/y_train.close
    test_baseline_equity_results['pct_return_achieved'] = test_baseline_equity_results['return_achieved']/y_test.close

    return train_baseline_equity_results, test_baseline_equity_results
    
def get_equity_curve(equity_curve, test_equity_results, show_plot = False):
    """ Gets a cumulative an equity curve of results from test """
    test_equity = pd.Series(test_equity_results.return_achieved.cumsum(),name = 'equity')
    
    equity_curve = pd.concat([equity_curve, test_equity])
    
    # equity_curve = equity_curve.rename(columns={0:'equity'})
    
    return equity_curve
    
def perform_rolling_classification_modeling(features_to_use, features_to_scale, target, models_to_test, train_X_sets, test_X_sets, train_y_sets, test_y_sets):
    """ Iterate through classification models defined to produce predictions
    
    Return cross validation results as a DataFrame and equity plots data (dict)"""
    
    all_classification_dataset_model_results = pd.DataFrame()
    all_baseline_results = pd.DataFrame()

    baseline_equity_curve = pd.DataFrame()
    equity_plots_data = {}

    # Iterate through each model
    for model_to_test in models_to_test:
        print(f"Model under test:{model_to_test}", end = '\n')
        equity_curve = pd.DataFrame()
        # iterate through each train/test split in the data
        for dataset_number in range(len(train_X_sets)):

            print(f"Train length: {len(train_X_sets[dataset_number])}, Data split: {dataset_number+1}/{len(train_X_sets)}    ", end = '\r')

            X_train = train_X_sets[dataset_number]
            X_test = test_X_sets[dataset_number]

            y_train = train_y_sets[dataset_number]
            y_test = test_y_sets[dataset_number]

            if model_to_test == 'baseline':
                # Test baseline
                baseline_selection, baseline_train_accuracy, baseline_test_accuracy = test_class_baseline(y_train,
                                                                                                          y_test,
                                                                                                          target,
                                                                                                          set_number = dataset_number+1, 
                                                                                                          print_results = False)

                train_baseline_equity_results, test_baseline_equity_results = get_class_baseline_equity_results(y_train, 
                                                                                                               y_test, 
                                                                                                               baseline_selection)
                # Consolidate the baseline results from each split
                all_baseline_results = consolidate_baseline_results(all_baseline_results,
                                                                dataset_number,
                                                                baseline_train_accuracy,
                                                                baseline_test_accuracy,
                                                               train_baseline_equity_results, 
                                                                test_baseline_equity_results)

                baseline_equity_curve = get_equity_curve(baseline_equity_curve, test_baseline_equity_results)

                continue

            # Train and test using model, output classification reports and predictions
            train_classification_report, test_classification_report, train_predictions, test_predictions = train_and_test_dataset(model_to_test,
                                   X_train, 
                                   X_test, 
                                   y_train, 
                                   y_test, 
                                   target, print_results = False)

            train_equity_results, test_equity_results = get_equity_results(y_train, 
                                                                           y_test,  
                                                                           train_predictions, 
                                                                           test_predictions)


            # Consolidate the classification modeling results
            all_classification_dataset_model_results = consolidate_class_model_results(dataset_number,
                                                                               all_classification_dataset_model_results,
                                                                         train_classification_report,
                                                                         test_classification_report,
                                                                                      train_equity_results,
                                                                                      test_equity_results)
            # Generate equity curve of the test sets
            equity_curve= get_equity_curve(equity_curve, test_equity_results)

        if model_to_test == 'baseline':
            equity_plots_data[model_to_test] = baseline_equity_curve
        else:                                                 
            equity_plots_data[str(model_to_test)] = equity_curve

        # plot equity curve of all test splits for each model
        # equity_curve.plot()
        # plt.title(f'Equity Curve: {model_to_test}')
        # plt.ylabel('Equity ($)')

    # plot baseline equity curve for test splits
    # baseline_equity_curve.plot(legend=False)
    # plt.title(f'Baseline Equity Curve')
    # plt.ylabel('Equity ($)')


    average_cross_validate_result = consolidate_datasets(all_baseline_results, 
                                                         all_classification_dataset_model_results)
    
    return average_cross_validate_result, equity_plots_data



def test_final_model(features_to_use, features_to_scale, model_to_test, target, other_targets, df, test_size):
    """ Test the best model on the withheld test set, which here consists of the last <test_size> days 
    
    Returns: dataframe with equity results from applying this model to test and classification report
    
    Requires as inputs:
    features to use: list
    features to scale: list
    model_to_test: scikit-learn model object
    """

    # Use withheld test set on final model
    X_final_test = df[features_to_scale].iloc[-test_size:]
    y_final_test = df[[target,'fwd_ret','close']].iloc[-test_size:]

    # Use preceding X_train_length datapoints (2620) for training the final model
    X_final_train = df[features_to_scale].iloc[-(2620+test_size):-test_size]
    y_final_train = df[[target,'fwd_ret','close']].iloc[-(2620+test_size):-test_size]

    # Perform scaling after the split so scaler can be fit to each train set individually
    X_final_train_scaled, X_final_test_scaled = scale_X(X_final_train, X_final_test, features_to_use, features_to_scale, scaler_type = StandardScaler())

    # Train and test using model, output classification reports and predictions
    train_classification_report, test_classification_report, train_predictions, test_predictions = train_and_test_dataset(model_to_test,
                           X_final_train_scaled, 
                           X_final_test_scaled, 
                           y_final_train, 
                           y_final_test, 
                           target, print_results = False)

    train_equity_results, final_test_equity_results = get_equity_results(y_final_train, 
                                                                   y_final_test,  
                                                                   train_predictions, 
                                                                   test_predictions)


    final_test_equity_curve = pd.DataFrame()
    # Generate equity curve of the test sets
    final_test_equity_curve= get_equity_curve(final_test_equity_curve, final_test_equity_results)
    
    return final_test_equity_curve, test_classification_report


def plot_final_equity(final_model_validate_test_equity, df, test_size):
    """ Plot performance of final model on the withheld test set """
    
    fig, ax = plt.subplots(figsize = (10,6))
    ax.plot(final_model_validate_test_equity)
    ax.set_title('Best Model Equity Performance - 2022', fontsize = 18)
    ax.set_ylabel('Equity ($)', fontsize = 18)

    final_test_start = df.index.max()-timedelta(days = test_size)
    final_test_end = df.index.max()

    # ax.axvspan(train_start, train_end, label = "train", color = "green", alpha = 0.3)
    ax.axvspan(final_test_start, final_test_end, label = "final test set", color = "green", alpha = 0.3)
    ax.annotate('Final Test Set', (mdates.datestr2num('2022-04-25'), 10000), fontsize = 15)
    ax.xaxis.set_major_formatter(DateFormatter('%B'))