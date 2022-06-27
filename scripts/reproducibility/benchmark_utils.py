import numpy as np
import pandas as pd
import csv
import io
import os
import multiprocessing as mp
import gc
import random
import joblib
import json
import re
import subprocess
import joblib
import datetime
import glob

import matplotlib as mpl
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import roc_auc_score, roc_curve, recall_score, precision_recall_curve

mpl.rcParams['figure.dpi'] = 150
pd.set_option('display.max_columns', 500)
pd.set_option('display.max_rows', 500)
pd.set_option('display.width', 200)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

import logging
FORMAT = "%(levelname)s: %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.WARN, format=FORMAT, datefmt=DATE_FORMAT)
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

from autogluon.tabular import TabularPredictor

import h2o
from h2o.automl import H2OAutoML

def load_data(dataset, base_path):
    logger.info(dataset)
    
    df_train = pd.read_csv(f"{base_path}/{dataset}/train.csv", lineterminator='\n')
    logger.info(df_train.shape)
    
    df_test = pd.read_csv(f"{base_path}/{dataset}/test.csv")
    logger.info(df_test.shape)
    
    df_test_labels = pd.read_csv(f"{base_path}/{dataset}/test_labels.csv")
    logger.info(df_test_labels.shape)
    
    df_test = df_test.merge(df_test_labels, how="inner", on="EVENT_ID")
    logger.info(df_test.shape)
    
    
    features_to_exclude = ("EVENT_LABEL", "EVENT_TIMESTAMP", "LABEL_TIMESTAMP", "ENTITY_TYPE", "ENTITY_ID", "EVENT_ID")
    features = [x for x in df_test.columns if x not in features_to_exclude ]
    logger.info(len(features))
    logger.info(features)
    
    return features, df_train, df_test

def get_recall(fpr, tpr, fpr_target=0.01): 
    return np.interp(fpr_target, fpr, tpr)