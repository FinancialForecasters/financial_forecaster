from imports import *
from sklearn.metrics import mean_squared_error

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

