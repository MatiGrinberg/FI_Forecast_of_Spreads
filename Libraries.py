from bs4 import BeautifulSoup
from collections import Counter
from datetime import datetime, timedelta
from IPython.display import display, HTML
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense,LeakyReLU,LSTM
from tensorflow.keras.models import Sequential
from openpyxl.utils.dataframe import dataframe_to_rows
from scipy.spatial import distance
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr,percentileofscore
from sklearn import metrics,preprocessing,svm
from sklearn.cluster import DBSCAN,KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,GradientBoostingClassifier,GradientBoostingRegressor,IsolationForest
from sklearn.linear_model import Lasso, Ridge,LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,mean_absolute_error,mean_squared_error,roc_auc_score, confusion_matrix, auc,roc_curve
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestNeighbors,KNeighborsRegressor,KNeighborsClassifier
from sklearn.preprocessing import StandardScaler,MinMaxScaler,OneHotEncoder,PolynomialFeatures
from sklearn.svm import SVR,SVC
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
#from request_html import HTMLSession
import datetime
import math
import matplotlib.pyplot as plt
import numpy as np
import openpyxl
import os
import pandas as pd
import pickle
import random
import seaborn as sns
from docx import Document
from docx.shared import Pt,Inches
#import shap
#import statsmodels.api as sm
import statistics as s 
import time
import xlsxwriter
from datetime import timedelta as td,datetime as dt
import dateutil.parser
from gym import spaces, Env
from gym.utils import seeding
from gym.envs.registration import register
from stable_baselines3 import A2C, SAC , PPO # , DDPG 
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy


