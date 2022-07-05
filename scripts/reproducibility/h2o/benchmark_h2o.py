import pandas as pd
import os
import gc
import joblib

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

import h2o
from h2o.automl import H2OAutoML
    
def run_h2o(dataset, base_path, connect_url=None, time_limit=None, include_algos=None, exclude_algos=None, verbosity="info", seed=10):
    if connect_url is not None:
        _ = h2o.connect(url=connect_url, https=True, verbose=True)
        h2o.cluster().show_status(True)
    else:
        h2o.init()
    
    gc.collect()
    features, df_train, df_test = load_data(dataset, base_path)
    
    df_train_h2o = h2o.H2OFrame(df_train)
    feature_types_h2o = {k:df_train_h2o.types[k] for k in df_train_h2o.types if k in features}
    # force test schema the same as train schema, otherwise predict will throw errors
    df_test_h2o = h2o.H2OFrame(df_test, column_types=feature_types_h2o)
    
    df_train_h2o['EVENT_LABEL'] = df_train_h2o['EVENT_LABEL'].asfactor()
    df_test_h2o['EVENT_LABEL'] = df_test_h2o['EVENT_LABEL'].asfactor()
        
    aml = H2OAutoML(max_runtime_secs = time_limit, seed = seed,
                     include_algos=include_algos,
                     exclude_algos=exclude_algos,
                 export_checkpoints_dir=f"{base_path}/{dataset}/H2OModels/",
                 verbosity=verbosity)
    
    # use validation error in the leaderboard to avoid leakage when calling aml.predict
    aml.train(x = features, 
          y = 'EVENT_LABEL', 
          training_frame = df_train_h2o,  
             )
    
    lb = aml.leaderboard
    # lb.head(rows=lb.nrows)
    
    h2o.h2o.download_csv(lb, f"{base_path}/{dataset}/leaderboard_h2o.csv")
    
    lb_2 = h2o.automl.get_leaderboard(aml, extra_columns = "ALL")
    h2o.h2o.download_csv(lb_2, f"{base_path}/{dataset}/leaderboard_h2o_full.csv")
    # Get training timing info
    info = aml.training_info
    joblib.dump(info, f"{base_path}/{dataset}/training_info.joblib")
    
    df_pred_h2o = aml.predict(df_test_h2o[features])
    pos_label = df_test_h2o['EVENT_LABEL'].levels()[0][-1] # levels are ordered alphabetically

    pos_label2 = 'p'+pos_label if pos_label=='1' else pos_label
    df_pred_h2o = (h2o.as_list(df_pred_h2o[pos_label2]))[pos_label2]

    auc = roc_auc_score(df_test['EVENT_LABEL'], df_pred_h2o)
    logger.info(f"auc on test data: {auc}")
    
    fpr, tpr, thresholds = roc_curve(df_test['EVENT_LABEL'].astype(str), df_pred_h2o, 
                                     pos_label=pos_label)
    
    y_true = df_test['EVENT_LABEL']
    y_true = (y_true.astype(str)==pos_label)
    
    recall = get_recall(fpr, tpr, fpr_target=0.01)
    logger.info(f"tpr@1%fpr on test data: {recall}")

    test_metrics_h2o = {
    "pos_label": pos_label,
    "labels": df_test['EVENT_LABEL'],
    "pred_prob": df_pred_h2o,    
    "auc": auc,
    "tpr@1%fpr": recall,
    "fpr": fpr,
    "tpr": tpr,
    "thresholds": thresholds
    }
    joblib.dump(test_metrics_h2o, f"{base_path}/{dataset}/test_metrics_h2o.joblib")
    
    h2o.cluster().shutdown(prompt=False)