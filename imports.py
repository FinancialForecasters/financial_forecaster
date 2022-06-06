# imports.py

# local-host
import unicodedata, itertools, re, requests, math, random, os, datetime, json, pprint, sys

# python data science library's
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import Holt
from scipy import stats
# from pydataset import data


# sci-kit-learn modules
from sklearn import metrics 
	# (
	# confusion_matrix,accuracy_score,precision_score,recall_score,
	# classification_report,mean_squared_error,r2_score,explained_variance_score
	# )
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, RFE, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression, LassoLars, TweedieRegressor
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.naive_bayes import BernoulliNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, PolynomialFeatures
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text


# visualizations
from pandas.plotting import register_matplotlib_converters

# matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib import style
from mpl_toolkits.mplot3d import Axes3D

# seaborn
import seaborn as sns

# talib
import talib

# beautifulsoup
from bs4 import BeautifulSoup

# snsscrape
# import snscrape.modules.twitter as sntwitter

# datetime
from datetime import datetime, timedelta


# natural language toolkit
import nltk
import nltk.sentiment
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords

# state properties
np.random.seed(123)
# pd.set_option("display.max_columns", None)
# pd.set_option("display.max_rows", None)