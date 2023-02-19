# FDB: Fraud Dataset Benchmark

*By [Prince Grover](groverpr), [Zheng Li](zhengli0817), Jianbo Liu, [Jakub Zablocki](qbaza), [Hao Zhou](haozhouamzn), Julia Xu and Anqi Cheng*


[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg?color=purple)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 



The **Fraud Dataset Benchmark (FDB)** is a compilation of publicly available datasets relevant to **fraud detection** ([arXiv Link](https://arxiv.org/abs/2208.14417)). The FDB aims to cover a wide variety of fraud detection tasks, ranging from card not present transaction fraud, bot attacks, malicious traffic, loan risk and content moderation. The Python based data loaders from FDB provide dataset loading, standardized train-test splits and performance evaluation metrics. The goal of our work is to provide researchers working in the field of fraud and abuse detection a standardized set of benchmarking datasets and evaluation tools for their experiments. Using FDB tools we evaluate 4 AutoML pipelines including AutoGluon, H2O, Amazon Fraud Detector and Auto-sklearn across 9 different fraud detection datasets and discuss the results. 


## Datasets used in FDB
Brief summary of the datasets used in FDB. Each dataset is described in detail in [data source section](#data-sources).

| **#** | **Dataset name**                                           | **Dataset key** | **Fraud category**                  | **#Train** | **#Test** | **Class ratio (train)** | **#Feats** | **#Cat** | **#Num** | **#Text** | **#Enrichable** |
|-------|------------------------------------------------------------|-----------------|-------------------------------------|------------|-----------|-------------------------|------------|----------|----------|-----------|-----------------|
| 1     | IEEE-CIS Fraud Detection                                   | ieeecis         | Card Not Present Transactions Fraud | 561,013    | 28,527    | 3.50%                   | 67         | 6        | 61       | 0         | 0               |
| 2     | Credit Card Fraud Detection                                | ccfraud         | Card Not Present Transactions Fraud | 227,845    | 56,962    | 0.18%                   | 28         | 0        | 28       | 0         | 0               |
| 3     | Fraud ecommerce                                            | fraudecom       | Card Not Present Transactions Fraud | 120,889    | 30,223    | 10.60%                  | 6          | 2        | 3        | 0         | 1               |
| 4     | Simulated Credit Card Transactions generated using Sparkov | sparknov        | Card Not Present Transactions Fraud | 1,296,675  | 20,000    | 5.70%                   | 17         | 10       | 6        | 1         | 0               |
| 5     | Twitter Bots Accounts                                      | twitterbot      | Bot Attacks                         | 29,950     | 7,488     | 33.10%                  | 16         | 6        | 6        | 4         | 0               |
| 6     | Malicious URLs dataset                                     | malurl          | Malicious Traffic                  | 586,072   | 65,119    | 34.20%                  | 2          | 0        | 1        | 1         | 0               |
| 7     | Fake Job Posting Prediction                                | fakejob         | Content Moderation                  | 14,304     | 3,576     | 4.70%                   | 16         | 10       | 1        | 5         | 0               |
| 8     | Vehicle Loan Default Prediction                            | vehicleloan    | Credit Risk                         | 186,523    | 46,631    | 21.60%                  | 38         | 13       | 22       | 3         | 0               |
| 9     | IP Blocklist                                               | ipblock         | Malicious Traffic                   | 172,000    | 43,000    | 7%                      | 1          | 0        | 0        | 0         | 1               |


## Installation

### Requirements
- Kaggle account
    - **Important**: `ieeecis` dataset requires you to [**join IEEE-CIS competetion**](https://www.kaggle.com/competitions/ieee-fraud-detection/overview) from your Kaggle account, before you can call fdb API. Otherwise you will get <span style="color:red">ApiException: (403)</span>.
- AWS account
- Python 3.7+ 

- Python requirements
```
autogluon==0.4.2
h2o==3.36.1.2
boto3==1.20.21
click==8.0.3
click-plugins==1.1.1
Faker==4.14.2
joblib==1.0.0
kaggle==1.5.12
numpy==1.19.5
pandas==1.1.2
regex==2020.7.14
scikit-learn==0.22.1
scipy==1.5.4
auto-sklearn==0.14.7
dask==2022.8.1
```

### Step 1: Setup Kaggle CLI
The `FraudDatasetBenchmark` object is going to load datasets from the source (which in most of the cases is Kaggle), and then it will modify/standardize on the fly, and provide train-test splits. So, the first step is to setup Kaggle CLI in the machine being used to run Python.

Use intructions from [How to Use Kaggle](https://www.kaggle.com/docs/api) guide. The steps include:

Remember to download the authentication token from "My Account" on Kaggle, and save token at `~/.kaggle/kaggle.json` on Linux, OSX and at `C:\Users<Windows-username>.kaggle\kaggle.json` on Windows. If the token is not there, an error will be raised. Hence, once you’ve downloaded the token, you should move it from your Downloads folder to this folder.
  
    
#### Step 1.2. [Join IEEE-CIS competetion](https://www.kaggle.com/competitions/ieee-fraud-detection/overview) from your Kaggle account, before you can call `fdb.datasets` with `ieeecis`. Otherwise you will get <span style="color:red">ApiException: (403)</span>.
  
  
### Step 2: Clone Repo
Once Kaggle CLI is setup and installed, clone the github repo using `git clone https://github.com/amazon-research/fraud-dataset-benchmark.git` if using HTTPS, or `git clone git@github.com:amazon-research/fraud-dataset-benchmark.git` if using SSH. 

### Step 3: Install
Once repo is cloned, from your terminal, `cd` to the repo and type `pip install .`, which will install the required classes and methods.


## FraudDatasetBenchmark Usage
The usage is straightforward, where you create a `dataset` object of `FraudDatasetBenchmark` class, and extract useful goodies like train/test splits and eval_metrics.   

**Important note**: If you are running multiple experiments that require re-loading dataframes multiple times, default setting of downloading from Kaggle before loading into dataframe exceed the account level API limits. So, use the setting to persist the downloaded dataset and then load from the persisted data. During the first call of FraudDatasetBenchmark(), use `load_pre_downloaded=False, delete_downloaded=False` and for subsequent calls, use `load_pre_downloaded=True, delete_downloaded=False`. The default setting is 
`load_pre_downloaded=False, delete_downloaded=True`
```
from fdb.datasets import FraudDatasetBenchmark

# all_keys = ['fakejob', 'vehicleloan', 'malurl', 'ieeecis', 'ccfraud', 'fraudecom', 'twitterbot', 'ipblock'] 
key = 'ipblock'

obj = FraudDatasetBenchmark(
    key=key,
    load_pre_downloaded=False,  # default
    delete_downloaded=True,  # default
    add_random_values_if_real_na = { 
        "EVENT_TIMESTAMP": True, 
        "LABEL_TIMESTAMP": True,
        "ENTITY_ID": True,
        "ENTITY_TYPE": True,
        "ENTITY_ID": True,
        "EVENT_ID": True
        } # default
    )
print(obj.key)

print('Train set: ')
display(obj.train.head())
print(len(obj.train.columns))
print(obj.train.shape)

print('Test set: ')
display(obj.test.head())
print(obj.test.shape)

print('Test scores')
display(obj.test_labels.head())
print(obj.test_labels['EVENT_LABEL'].value_counts())
print(obj.train['EVENT_LABEL'].value_counts(normalize=True))
print('=========')

``` 
Notebook template to load dataset using FDB data-loader is available at [scripts/examples/Test_FDB_Loader.ipynb](scripts/examples/Test_FDB_Loader.ipynb)

## Reproducibility
Reproducibility scripts are available at [scripts/reproducibility/](scripts/reproducibility/) in respective folders for [afd](scripts/reproducibility/afd), [autogluon](scripts/reproducibility/autogluon) and [h2o](scripts/reproducibility/h2o). Each folder also had README with steps to reproduce.


## Benchmark Results

<!-- | **Dataset key** | **AUC-ROC** |             |               |                  |                  | **Recall at 1% FPR** |             |               |                  |                  |
|:---------------:|:-----------:|:-----------:|:-------------:|:----------------:|:----------------:|:--------------------:|:-----------:|:-------------:|:----------------:|:----------------:|
|                 | **AFD OFI** | **AFD TFI** | **AutoGluon** |      **H2O**     | **Auto-sklearn** |      **AFD OFI**     | **AFD TFI** | **AutoGluon** |      **H2O**     | **Auto-sklearn** |
|     ccfraud     |    0.985    |     0.99    |      0.99     |     **0.992**    |       0.988      |         0.88         |     0.88    |      0.88     |       0.853      |       0.88       |
|     fakejob     |    0.987    |      -      |   **0.998**   |       0.99       |       0.983      |         0.786        |      -      |     0.925     |       0.781      |       0.781      |
|    fraudecom    |    0.519    |  **0.636**  |     0.522     |       0.518      |       0.515      |         0.011        |    0.099    |     0.012     |       0.009      |       0.012      |
|     ieeecis     |    0.938    |   **0.94**  |     0.855     |       0.89       |       0.932      |         0.587        |     0.56    |     0.425     |       0.442      |       0.569      |
|      malurl     |    0.985    |      -      |   **0.998**   | Training failure |        0.5       |         0.868        |      -      |     0.976     | Training failure |       0.01       |
|     sparknov    |  **0.998**  |      -      |     0.997     |       0.997      |       0.995      |           1          |      -      |     0.927     |       0.896      |       0.868      |
|    twitterbot   |    0.934    |      -      |   **0.943**   |       0.938      |       0.936      |         0.518        |      -      |     0.419     |       0.382      |       0.369      |
|   vehicleloan   |  **0.673**  |      -      |     0.669     |       0.67       |       0.664      |         0.036        |      -      |      0.04     |       0.037      |       0.035      |
|     ipblock     |  **0.937**  |      -      |     0.804     | Training failure |        0.5       |         0.466        |      -      |      0.32     | Training failure |       0.01       | -->

| **Dataset key** | **AUC-ROC** |             |               |                  |                  |
|:---------------:|:-----------:|:-----------:|:-------------:|:----------------:|:----------------:|
|                 | **AFD OFI** | **AFD TFI** | **AutoGluon** |      **H2O**     | **Auto-sklearn** |
|     ccfraud     |    0.985    |     0.99    |      0.99     |     **0.992**    |       0.988      |
|     fakejob     |    0.987    |      -      |   **0.998**   |       0.99       |       0.983      |
|    fraudecom    |    0.519    |  **0.636**  |     0.522     |       0.518      |       0.515      |
|     ieeecis     |    0.938    |   **0.94**  |     0.855     |       0.89       |       0.932      |
|      malurl     |    0.985    |      -      |   **0.998**   | Training failure |        0.5       |
|     sparknov    |  **0.998**  |      -      |     0.997     |       0.997      |       0.995      |
|    twitterbot   |    0.934    |      -      |   **0.943**   |       0.938      |       0.936      |
|   vehicleloan   |  **0.673**  |      -      |     0.669     |       0.67       |       0.664      |
|     ipblock     |  **0.937**  |      -      |     0.804     | Training failure |        0.5       |

### ROC Curves

The numbers in the legend represent AUC-ROC from different models from our baseline evaluations on AutoML.  
![roc curves](images/all_fdb.png)


## Data Sources

1. **IEEE-CIS Fraud Detection**
    - Link: https://www.kaggle.com/c/ieee-fraud-detection/overview
    - Feature info: Card, address, email, product id, aggregates
    - Fraud category: Card Not Present Transaction Fraud
    - Provider: [Vesta Corporation](https://www.vesta.io/)

2. **Credit Card Fraud Detection**
    - Link: https://www.kaggle.com/mlg-ulb/creditcardfraud/
    - Feature info: PCA features, time, amount (highly imbalanced)
    - Fraud category: Card Not Present Transaction Fraud
    - Provider: [Machine Learning Group - ULB](https://mlg.ulb.ac.be/)

3. **Fraud ecommerce**
    - Link: https://www.kaggle.com/vbinh002/fraud-ecommerce
    - Feature info: Signup time, purchase time, purchase value, ip, browser, age
    - Fraud category: Card Not Present Transaction Fraud
    - Provider: [Binh Vu](https://www.kaggle.com/vbinh002) 

4. **Simulated Credit Card Transactions generated using Sparkov**
    - Link: https://www.kaggle.com/kartik2112/fraud-detection
    - Feature info: Cc_num, merchant, txn_date, category, zip, location
    - Fraud category: Card Not Present Transaction Fraud
    - Provider: [Kartik Shenoy](https://www.kaggle.com/kartik2112) 

5. **Twitter Bots Accounts**
    - Link: https://www.kaggle.com/code/davidmartngutirrez/bots-accounts-eda/data?select=twitter_human_bots_dataset.csv
    - Feature info: Followers/following count, geo-enabled, description etc.
    - Fraud category: Bot Attacks
    - Provider: [David Martín Gutiérrez](https://www.kaggle.com/davidmartngutirrez) 

6. **Malicious URLs dataset**
    - Link: https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset
    - Feature info: URL (malicious include defacement, phishing and malware)
    - Fraud category: Malicious Traffic
    - Provider: [Manu Siddhartha](https://www.kaggle.com/sid321axn) 

7. **Real / Fake Job Posting Prediction**
    - Link: https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction
    - Feature info: Textual information and meta-information about the jobs
    - Fraud category: Content Moderation
    - Provider: [Shivam Bansal](https://www.kaggle.com/shivamb) 

8. **Vehicle Loan Default Prediction**
    - Link: https://www.kaggle.com/avikpaul4u/vehicle-loan-default-prediction
    - Feature info: numeric, categorical, classification(binary)
    - Fraud category: Credit Risk
    - Provider: [Avik Paul](https://www.kaggle.com/avikpaul4u) 

9. **IP Blocklist**
    - Link: http://cinsscore.com/list/ci-badguys.txt
    - Feature info: Malicious IP address 
    - Fraud category: Malicious Traffic
    - Provider: [CINSscore.com](http://cinsscore.com)

## Citation
```
@misc{grover2022fdb,
      title={FDB: Fraud Dataset Benchmark}, 
      author={Prince Grover and Zheng Li and Jianbo Liu and Jakub Zablocki and Hao Zhou and Julia Xu and Anqi Cheng},
      year={2022},
      eprint={2208.14417},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## License
This project is licensed under the MIT-0 License.


## Acknowledgement
We thank creators of all datasets used in the benchmark and organizations that have helped in hosting the datasets and making them widely availabel for research purposes. 





