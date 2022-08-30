
import json
import joblib
import datetime
import numpy as np
import pandas as pd
import os, sys, shutil

from autosklearn.metrics import roc_auc, log_loss
from autosklearn.classification import AutoSklearnClassifier

from sklearn.metrics import roc_auc_score, roc_curve
from pandas.api.types import is_numeric_dtype, is_string_dtype

import logging
FORMAT = "%(levelname)s: %(name)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
logging.basicConfig(level=logging.WARN, format=FORMAT, datefmt=DATE_FORMAT)
logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

logging_config = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'simple': {
            'format': '%(levelname)-8s %(name)-15s %(message)s'
        }
    },
    'handlers':{
        'console_handler': {
            'class': 'logging.StreamHandler',
            'formatter': 'simple'
        },
        'file_handler': {
            'class':'logging.FileHandler',
            'mode': 'a',
            'encoding': 'utf-8',
            'filename':'main.log',
            'formatter': 'simple'
        },
        'spec_handler':{
            'class':'logging.FileHandler',
            'filename':'dummy_autosklearn.log',
            'formatter': 'simple'
        },
        'distributed_logfile':{
            'filename':'distributed.log',
            'class': 'logging.FileHandler',
            'formatter': 'simple',
            'level': 'DEBUG'
        }
    },
    'loggers': {
        '': {
            'level': 'INFO',
            'handlers':['file_handler', 'console_handler']
        },
        'autosklearn': {
            'level': 'INFO',
            'propagate': False,
            'handlers': ['spec_handler']
        },
        'smac': {
            'level': 'INFO',
            'propagate': False,
            'handlers': ['spec_handler']
        },
        'EnsembleBuilder': {
            'level': 'INFO',
            'propagate': False,
            'handlers': ['spec_handler']
        },
    },
}

def load_data(dataset_path):
    logger.info(dataset_path)
    
    df_train = pd.read_csv(f"{dataset_path}/train.csv", lineterminator='\n')
    logger.info(df_train.shape)
    
    df_test = pd.read_csv(f"{dataset_path}/test.csv")
    logger.info(df_test.shape)
    
    df_test_labels = pd.read_csv(f"{dataset_path}/test_labels.csv")
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


def run_autosklearn(dataset_path):
    
    features, df_train, df_test = load_data(dataset_path)

    dateTimeObj = datetime.datetime.now()
    timestampStr = dateTimeObj.strftime("%Y%m%d_%H%M%S")
    
    numeric_features = [f for f in features if is_numeric_dtype(df_train[f])]
    categorical_features = [f for f in features if f not in numeric_features]
    logger.info(f'categorical: {categorical_features}')
    logger.info(f'numeric: {numeric_features}')
    
    labels = sorted(df_train['EVENT_LABEL'].unique())
    df_train['EVENT_LABEL'].replace({labels[0]: 0, labels[1]: 1}, inplace=True)
    df_test['EVENT_LABEL'].replace({labels[0]: 0, labels[1]: 1}, inplace=True)
    
    for df in [df_train, df_test]:
        df[categorical_features] = df[categorical_features].fillna('<nan>')
        df[categorical_features] = df[categorical_features].astype('category')
    
    out_dir = f"{dataset_path}/AutoSklearnModels/"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    
    automl = AutoSklearnClassifier(
        metric=roc_auc,
        scoring_functions=[roc_auc, log_loss],
        tmp_folder=out_dir, # for debugging
        delete_tmp_folder_after_terminate=False,
        logging_config=logging_config,
        n_jobs=-1,
        memory_limit=None
    )
    
    assert len(categorical_features) + len(numeric_features) == len(features)
    
    logger.info('Fitting')
    automl.fit(df_train[features], df_train['EVENT_LABEL'])
    joblib.dump(automl, f"{dataset_path}/automl.joblib")
    
    cv = pd.DataFrame(automl.cv_results_)
    cv.to_csv(f"{dataset_path}/cv_results_autosklearn.csv", index=False)
    
    df_pred = automl.predict_proba(df_test[features])[:,1]
    
    auc_score = roc_auc_score(df_test['EVENT_LABEL'], df_pred)
    logger.info(f"auc on test data: {auc_score}")
    
    fpr, tpr, thresholds = roc_curve(df_test['EVENT_LABEL'], df_pred)
        
    recall = get_recall(fpr, tpr, fpr_target=0.01)
    logger.info(f"tpr@1%fpr on test data: {recall}")
    
    test_metrics = {
    "labels": df_test['EVENT_LABEL'],
    "pred_prob": df_pred,    
    "auc": auc_score,
    "tpr@1%fpr": recall,
    "fpr": fpr,
    "tpr": tpr,
    "thresholds": thresholds
    }
    joblib.dump(test_metrics, f"{dataset_path}/test_metrics_autosklearn.joblib")
    
if __name__ == "__main__":
    args = sys.argv
    logger.info(args)
    run_autosklearn(args[1])
    