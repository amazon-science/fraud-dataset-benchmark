## Steps to reproduce Auto-sklearn models


1. Load and save the datasets locally using [FDB Loader](../../examples/Test_FDB_Loader.ipynb). Keep note of `{DATASET_PATH}` that contains local paths to datasets containing `train.csv`, `test.csv` and `test_labels.csv` from FDB loader.

2. Run `benchmark_autosklearn.py` using following:
```
python3 benchmark_autosklearn.py {DATASET_PATH}
```

3. The script after running successfully will save results in the `DATASET_PATH`. The evaluation metrics on `test.csv` will be saved in `test_metrics_autosklearn.joblib`. 

*Note: Python 3.7+ is needed to run the used version of auto-sklearn and to reproduce the results. Similar to other auto-ml frameworks, auto-sklearn is also not perfectly reproducible because some underlying models are not deterministically seeded. However, the variations in results are within acceptable errors.*
