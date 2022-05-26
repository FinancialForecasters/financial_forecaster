# env
import sys
local_path = '/Users/hinzlehome/codeup-data-science/binance-project/'
sys.path.insert(0,local_path+'.env')
from env import *


# local-host
import math, os, datetime, json, pprint, requests
# having a weird time requests


# python data science library's
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.tsa.api import Holt

# sci-kit-learn modules
from sklearnex import patch_sklearn
patch_sklearn()
# $ conda install scikit-learn-intelex
# $ python -m sklearnex my_application.py
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, RFE, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoLars, TweedieRegressor
from sklearn.metrics import (
	confusion_matrix,accuracy_score,precision_score,recall_score,
	classification_report,mean_squared_error,r2_score,explained_variance_score
	)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text


# visualizations
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
from pandas.plotting import register_matplotlib_converters
from pydataset import data

# binance modules
import websocket, talib
from binance.client import Client
from binance.enums import *

# facebook "prophet"
from prophet import Prophet

# state properties
np.random.seed(123)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)