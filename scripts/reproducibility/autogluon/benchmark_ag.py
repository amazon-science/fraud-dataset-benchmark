import pandas as pd
import os
import gc
import joblib
import datetime

import matplotlib as mpl
from sklearn.metrics import roc_auc_score, roc_curve

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

import sys
sys.path.append('../')
from benchmark_utils import load_data, get_recall

from autogluon.tabular import TabularPredictor

def run_ag(dataset, base_path, time_limit=3600, presets=None, hyperparameters=None, feature_metadata='infer', verbosity=2):
    gc.collect()
    features, df_train, df_test = load_data(dataset, base_path)

    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%Y%m%d_%H%M%S")
    
    suffix = (f"_{presets}" if presets is not None else "") \
            + (f"_{hyperparameters}" if hyperparameters is not None else "") \
            + ("_feature_metadata" if feature_metadata != 'infer' else "") 
    folder = f"ag-{timestampStr}" \
            + suffix

    predictor = TabularPredictor(label='EVENT_LABEL', eval_metric='roc_auc', path=f"{base_path}/{dataset}/AutogluonModels/{folder}/", 
                                 verbosity=verbosity)
    predictor.fit(df_train[features + ['EVENT_LABEL'] ], 
                  time_limit=time_limit, presets=presets, hyperparameters=hyperparameters, feature_metadata=feature_metadata)

    leaderboard = predictor.leaderboard(df_test[features + ['EVENT_LABEL'] ])

    leaderboard_file = "leaderboard" \
                        + suffix \
                        + ".csv"
    leaderboard.to_csv(f"{base_path}/{dataset}/{leaderboard_file}", index=False)
    
    df_pred = predictor.predict_proba(df_test[ features ], 
                                                            as_multiclass=False)
    
    auc = roc_auc_score(df_test['EVENT_LABEL'], df_pred)
    logger.info(f"auc on test data: {auc}")
    pos_label = predictor.positive_class
    fpr, tpr, thresholds = roc_curve(df_test['EVENT_LABEL'], df_pred, 
                                     pos_label=pos_label)
    
    y_true = df_test['EVENT_LABEL']
    y_true = (y_true==pos_label)
    
    recall = get_recall(fpr, tpr, fpr_target=0.01)
    logger.info(f"tpr@1%fpr on test data: {recall}")
    
    test_metrics_ag_bq = {
    "labels": df_test['EVENT_LABEL'],
    "pred_prob": df_pred,    
    "auc": auc,
    "tpr@1%fpr": recall,
    "fpr": fpr,
    "tpr": tpr,
    "thresholds": thresholds
    }
    metrics_file = "test_metrics_ag" \
                    + suffix \
                    + ".joblib"
    joblib.dump(test_metrics_ag_bq, f"{base_path}/{dataset}/{metrics_file}")