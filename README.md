# FDB: Fraud Dataset Benchmark

*By [Prince Grover](groverpr), [Zheng Li](zhengli0817), [Julia Xu](SheliaXin), [Justin Tittelfitz](jtittelfitz), Anqi Cheng, [Jakub Zablocki](qbaza), Jianbo Liu, and [Hao Zhou](haozhouamzn)*


[![made-with-python](https://img.shields.io/badge/Made%20with-Python-1f425f.svg?color=purple)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 


The **Fraud Dataset Benchmark (FDB)** is a compilation of publicly available datasets relevant to **fraud detection** ([arXiv Link](https://arxiv.org/abs/2208.14417)). The FDB aims to cover a wide variety of fraud detection tasks, ranging from card not present transaction fraud, bot attacks, malicious traffic, loan risk and content moderation. The Python based data loaders from FDB provide dataset loading, standardized train-test splits and performance evaluation metrics. The goal of our work is to provide researchers working in the field of fraud and abuse detection a standardized set of benchmarking datasets and evaluation tools for their experiments. Using FDB tools we We demonstrate several applications of FDB that are of broad interest for fraud detection, including feature engineering, comparison of supervised learning algorithms, label noise removal, class-imbalance treatment and semi-supervised learning. 


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
    - Source URL: https://www.kaggle.com/c/ieee-fraud-detection/overview
    - Source license: https://www.kaggle.com/competitions/ieee-fraud-detection/rules
    - Variables: Anonymized product, card, address, email domain, device, transaction date information. Numeric columns with name prefixes as V, C, D and M, and meaning hidden from public.
    - Fraud category: Card Not Present Transaction Fraud
    - Provider: [Vesta Corporation](https://www.vesta.io/)
    - Release date: 2019-10-03
    - Description: Prepared by IEEE Computational Intelligence Society, this card-non-present transaction fraud dataset was launched during IEEE-CIS Fraud Detection Kaggle competition, and was provided by Vesta Corporation. The original dataset contains 393 features which are reduced to 67 features in the benchmark. Feature selection was performed based on highly voted Kaggle kernels. The fraud rate in training segment of source dataset is 3.5%. We only used training files (train transaction and train identity) containing 590,540 transactions in the benchmark, and split that into train (95%) and test (5%) segments based on time. Based on the insights from a Kaggle kernel written by the competition winner, we added UUID (called it as ENTITY_ID) that represents a fingerprint and was created using card, address, time and D1 features.

2. **Credit Card Fraud Detection**
    - Source URL: https://www.kaggle.com/mlg-ulb/creditcardfraud/
    - Source license: https://opendatacommons.org/licenses/dbcl/1-0/
    - Variables: PCA transformed features, time, amount (highly imbalanced)
    - Fraud category: Card Not Present Transaction Fraud
    - Provider: [Machine Learning Group - ULB](https://mlg.ulb.ac.be/)
    - Release date: 2018-03-23
    - Description: This dataset contains anonymized credit card transactions by European cardholders in September 2013. The dataset contains 492 frauds out of 284,807 transactions over 2 days. Data only contains numerical features that are the result of a PCA transformation, plus non transformed time and amount.

3. **Fraud ecommerce**
    - Source URL: https://www.kaggle.com/vbinh002/fraud-ecommerce
    - Source license: None
    - Variables: The features include sign up time, purchase time, purchase value, device id, user id, browser, and IP address. We added a new feature that measured the time difference between sign up and purchase, as the age of an account is often an important variable in fraud detection.
    - Fraud category: Card Not Present Transaction Fraud
    - Provider: [Binh Vu](https://www.kaggle.com/vbinh002) 
    - Release date: 2018-12-09
    - Description: This dataset contains ~150k e-commerce transactions.

4. **Simulated Credit Card Transactions generated using Sparkov**
    - Source URL: https://www.kaggle.com/kartik2112/fraud-detection
    - Source license: https://creativecommons.org/publicdomain/zero/1.0/
    - Variables: Transaction date, credit card number, merchant, category, amount, name, street, gender. All variables are synthetically generated using the Sparknov tool.
    - Fraud category: Card Not Present Transaction Fraud
    - Provider: [Kartik Shenoy](https://www.kaggle.com/kartik2112)
    - Release date: 2020-08-05
    - Description: This is a simulated credit card transaction dataset. The dataset was generated using Sparkov Data Generation tool and we modified a version of dataset created for Kaggle. It covers transactions of 1000 customers with a pool of 800 merchants over 6 months. We used both train and test segments directly from the source and randomly down sampled test segment.

5. **Twitter Bots Accounts**
    - Source URL: https://www.kaggle.com/code/davidmartngutirrez/bots-accounts-eda/data?select=twitter_human_bots_dataset.csv
    - Source license: https://creativecommons.org/publicdomain/zero/1.0/
    - Variables: Features like account creation date, follower and following counts, profile description, account age, meta data about profile picture and account activity, and a label indicating whether the account is human or bot.
    - Fraud category: Bot Attacks
    - Provider: [David Martín Gutiérrez](https://www.kaggle.com/davidmartngutirrez)
    - Release date: 2020-08-20
    - Description: The dataset composes of 37,438 rows corresponding to different user accounts from Twitter.

6. **Malicious URLs dataset**
    - Source URL: https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset
    - Source license: https://creativecommons.org/publicdomain/zero/1.0/
    - Variables: The Kaggle dataset is curated using five different sources, and contains url and type. Even though original dataset has multiclass label (type), we converted it into binary label.
    - Fraud category: Malicious Traffic
    - Provider: [Manu Siddhartha](https://www.kaggle.com/sid321axn) 
    - Release date: 2021-07-23
    - Description: The Kaggle dataset is curated using five different sources, and contains url and type. Even though original dataset has multiclass label (type), we converted it into binary label. There is no timestamp information from the source. Therefore, we generate a dummy timestamp column for consistency.

7. **Real / Fake Job Posting Prediction**
    - Source URL: https://www.kaggle.com/shivamb/real-or-fake-fake-jobposting-prediction
    - Source license: https://creativecommons.org/publicdomain/zero/1.0/
    - Variables: Title, location, department, company, salary range, requirements, description, benefits, telecommuting. Most of the variables are categorical and free form text in nature.
    - Fraud category: Content Moderation
    - Provider: [Shivam Bansal](https://www.kaggle.com/shivamb) 
    - Release date: 2020-02-29
    - Description: This Kaggle dataset contains 18K job descriptions out of which about 800 are fake. The data consists of both textual information and meta-information about the jobs. The task is to train classification model to detect which job posts are fraudulent.

8. **Vehicle Loan Default Prediction**
    - Source URL: https://www.kaggle.com/avikpaul4u/vehicle-loan-default-prediction
    - Source license: Unknown
    - Variables: Loanee information, loan information, credit bureau data, and history.
    - Fraud category: Credit Risk
    - Provider: [Avik Paul](https://www.kaggle.com/avikpaul4u) 
    - Release date: 2019-11-12
    - Description: The task in this dataset is to determine the probability of vehicle loan default, particularly the risk of default on the first monthly installments. It contains data for 233k loans with 21.7% default rate.
    
9. **IP Blocklist**
    - Source URL: http://cinsscore.com/list/ci-badguys.txt
    - Source license: Unknown
    - Variables: The dataset contains IP address and label telling malicious or fake. A dummy categorical variable that has no relation label is added.
    - Fraud category: Malicious Traffic
    - Provider: [CINSscore.com](http://cinsscore.com)
    - Release date: 2017-09-25
    - Description: This dataset is made up from malicious IP address from cinsscore.com. To the list of malicious IP addresses, we added randomly generated IP address using Faker labeled as benign.
    

## Citation
```
@misc{grover2023fraud,
      title={Fraud Dataset Benchmark and Applications}, 
      author={Prince Grover and Julia Xu and Justin Tittelfitz and Anqi Cheng and Zheng Li and Jakub Zablocki and Jianbo Liu and Hao Zhou},
      year={2023},
      eprint={2208.14417},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## License
This project is licensed under the MIT-0 License.


## Acknowledgement
We thank creators of all datasets used in the benchmark and organizations that have helped in hosting the datasets and making them widely availabel for research purposes. 





