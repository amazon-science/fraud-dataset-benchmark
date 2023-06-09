import os
import re
import json
import pandas as pd
import numpy as np
import warnings
from datetime import datetime

from category_encoders.target_encoder import TargetEncoder
from skclean.simulate_noise import flip_labels_cc, BCNoise

from fdb.datasets import FraudDatasetBenchmark

import feature_dict

DATASET_PATH = './data/dataset.csv'
METADATA_PATH = './data/feature_metadata.json'
FD = feature_dict.feature_dict

def noise_amount(df):
    return df[df.noise == 1].shape[0]

def noise_rate(df):
    if df.shape[0] > 0:
        return noise_amount(df)/df.shape[0]
    else:
        return None

def type_1_noise_amount(df):
    # examples with true label 0, mislabeled as 1
    # here 'df.label' is the observed label, not the true one
    return df[(df.label==1) & (df.noise == 1)].shape[0]

def type_2_noise_amount(df):
    # examples with true label 1, mislabeled as 0
    # here 'df.label' is the observed label, not the true one
    return df[(df.label==0) & (df.noise == 1)].shape[0]

def actual_legit_amount(df):
    return df[(df.label == 0) | (df.noise == 1)].shape[0]

def observed_legit_amount(df):
    return df[df.label == 0].shape[0]

def actual_fraud_amount(df):
    return df[((df.label == 1) & (df.noise == 0)) | ((df.label == 0) & (df.noise == 1))].shape[0]

def observed_fraud_amount(df):
    return df[df.label == 1].shape[0]

def actual_fraud_rate(df):
    if df.shape[0] > 0:
        return actual_fraud_amount(df)/df.shape[0]
    else:
        return None

def observed_fraud_rate(df):
    if df.shape[0] > 0:
        return observed_fraud_amount(df)/df.shape[0]
    else:
        return None

def type_1_noise_rate(df):
    if df.shape[0] > 0:
        return type_1_noise_amount(df)/actual_legit_amount(df)
    else:
        return None

def type_2_noise_rate(df):
    if df.shape[0] > 0:
        return type_2_noise_amount(df)/actual_fraud_amount(df)
    else:
        return None

def prepare_data_fdb(key, drop_text_enr_features=True):    
    """
    main function, gets datasets from FDB and then does some preprocessing/cleaning so they are suitable
    for modeling, returns data and metadata
    
    inputs: 
        key - the FDB dataset to load
        drop_text_enr_features - whether we want to drop text/enrichable features
    this returns
        df - full pandas dataframe containing features, labels and metadata
            this includes training and test data, with a 'dataset' column to indicate which
            all of these datasets have a timestamp column (even if it is "fake") and by default
            data will be sorted by this column. All test > train w.r.t. this timestamp
            
        features - list of feature names
        cat_features - list of categorical feature names (subset of features)
        label - name of label column
        record_id - name of unique id column
    """
    
    obj = FraudDatasetBenchmark(key=key)
    
    print(obj.key)
    
    # extract training and testing data (and test labels) from the return object
    # sort training data by event timestamp
    train_df = obj.train.sort_values(by='EVENT_TIMESTAMP',ignore_index=True)
    test_df = obj.test.reset_index(drop=True)
    test_labels = obj.test_labels.reset_index(drop=True)

    # define metadata and label column names
    metadata = ['EVENT_LABEL', 'EVENT_TIMESTAMP', 'ENTITY_ID', 'ENTITY_TYPE', 'EVENT_ID',
                'label', 'LABEL_TIMESTAMP', 'noise', 'dataset']
    label = ['label']
    
    # we maintain a feature dictionary in another file, this helps us determine which are categorical, numerical, etc.
    feature_dict = FD[key]
    raw_features = feature_dict.keys()
    num_features = [f for f in raw_features if feature_dict[f] == 'numeric']
    cat_features = [f for f in raw_features if feature_dict[f] == 'categorical']
    txt_features = [f for f in raw_features if feature_dict[f] == 'text']
    enr_features = [f for f in raw_features if feature_dict[f] == 'enrichable']
    
    # add / rename labels
    train_df.rename({'EVENT_LABEL':'label'}, axis=1, inplace=True)
    test_df['label'] = test_labels['EVENT_LABEL']
    if key == 'twitterbot':
        train_df.loc[train_df.label == 'bot', 'label'] = 1
        test_df.loc[test_df.label == 'bot', 'label'] = 1
        train_df.loc[train_df.label == 'human', 'label'] = 0
        test_df.loc[test_df.label == 'human', 'label'] = 0

    # put train / test into single dataframe, create a 'dataset' column to keep track
    train_df['dataset'] = 'train'    
    test_df['dataset'] = 'test'

    # create noise column - we won't generate any noise now but it may be useful to have (can also be ignored)
    train_df['noise'] = 0
    test_df['noise'] = 0

    # concatenate train/test into single dataframe 
    # (remember we have 'dataset' column to separate them again if needed)
    df = pd.concat([train_df, test_df], axis=0, ignore_index=True)
    
    # there are a few date columns that are timestamps, we convert those to epoch
    # the new values are put into new columns, those column names are added to the numerical features
    if key == 'twitterbot':
        df['eng_created_at'] = df['created_at'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d %H:%M:%S').timestamp())
        num_features.append('eng_created_at')
    if key == 'sparknov':
        df['eng_dob'] = df['dob'].apply(lambda x : datetime.strptime(x, '%Y-%m-%d').timestamp())
        num_features.append('eng_dob')
    
    # fakejob has a salary range column, e.g. "10000 - 20000" that can be converted into two numerical columns
    if key == 'fakejob':
        def convert(x):
            r = re.search(r"([0-9]*)-([0-9]*)",str(x))
            try:
                m, M = r.group(1), r.group(2)
                if m == '' or M == '':
                    m, M = 0,0
            except:
                m, M = 0,0
            return m,M

        df['salary_min'], df['salary_max'] = zip(*df['salary_range'].map(convert))
        num_features = num_features + ['salary_min','salary_max']
    
    # vehicleloan has a timestamp column that we convert to epoch
    # it also has "account age" and "credit history" length cols 
    # in form "Xyrs Ymon" that can be converted to numeric
    if key == 'vehicleloan':
        df['eng_dob'] = df['date_of_birth'].apply(lambda x : datetime.strptime(x, '%d-%m-%Y').timestamp())
        
        def convert(x):
            r = re.search(r"([0-9]*)yrs ([0-9]*)mon", x)
            try:
                age = 12*float(r.group(1)) + float(r.group(2))
            except:
                age = 0
            return age
    
        df['eng_average_acct_age'] = df['average_acct_age'].apply(convert)
        df['eng_credit_history_length'] = df['credit_history_length'].apply(convert)
        num_features = num_features + ['eng_dob','eng_average_acct_age','eng_credit_history_length']
    
    # by default we will drop any remaining text or enrichable (IP address) features as we won't use them
    # but you can pass in False for this if they are of interest
    if drop_text_enr_features:
        df.drop(txt_features + enr_features, axis=1, inplace=True)
        features = num_features + cat_features
    
    # cast all numeric features to float just in case they aren't
    for feature in num_features:
        df[feature] = df[feature].astype('float64')
        df[feature].fillna(0, inplace=True)
    
    # cast all categorical features to str in case they aren't
    for feature in cat_features:
        df[feature] = df[feature].astype(str) 
        df[feature].fillna('', inplace=True)
    
    # rename the timestamp column
    df.rename({'EVENT_TIMESTAMP':'creation_date'}, axis=1, inplace=True)    
    
    # cast the label to int just to be sure
    df['label'] = df['label'].astype('int')
    
    # name of unique id column will always be EVENT_ID
    record_id = 'EVENT_ID'
    
    if drop_text_enr_features:
        return df, features, cat_features, label, record_id
    else:
        return df, features, cat_features, txt_features, enr_features, label, record_id


def add_noise(df, noise_type, noise_amount, *, time_index=None, features=None, cat_features=None, label=None):

    if noise_type not in ['random', 'time-dependent', 'boundary-consistent']:
        raise(Exception('Invalid Noise Type'))
    
    # if we want time-dependent noise it will be useful to convert timestamps into epoch
    def convert_to_millis(x):
        try:
            m = datetime.strptime(x, '%Y-%m-%dT%H:%M:%SZ').timestamp()
        except:
            m = datetime.strptime(x, '%Y-%m-%d %H:%M:%S').timestamp()
        return m

    # random noise can be class-conditional in both directions (other types of noise cannot)
    # if noise_amount is passed in as [r,s] we can flip labels in both directions: 
    #    r is percent of 0s flipped to 1s
    #    s is percent of 1s flipped to 0s
    # for random noise, if noise_amount is a single number, assume it is s, and that r=0 
    #   (i.e. class-conditional noise where only 1s get flipped to 0s)
    if isinstance(noise_amount, tuple) or isinstance(noise_amount, list): 
        if noise_type != 'random':
            raise(Exception('For time-dependent and boundary-consistent noise,'
                            'only a single value is allowed for noise_amount'))
        r = noise_amount[0]
        s = noise_amount[1]
    else:
        r = 0
        s = noise_amount
    
    # we will add noise to a *copy* of the dataframe
    df_copy = df.copy()
    
    if noise_type == 'time-dependent':
        df_copy['event_millis'] = df_copy[time_index].apply(convert_to_millis)
        df_copy['event_millis'] = df_copy['event_millis'] - df_copy['event_millis'].min()    
        mislabel = df_copy[(df_copy.noise == 0) 
                           & (df_copy.label == 1)].sample(frac = s, 
                                                                 weights=df_copy['event_millis']).index
        df_copy.loc[mislabel,'noise'] = 1
        df_copy.loc[mislabel,'label'] = 0
    else:
        if noise_type == 'boundary-consistent':
            from catboost import CatBoostClassifier
            warnings.filterwarnings("ignore", category=FutureWarning)
            target_encoder = TargetEncoder(cols=cat_features)
            reshaped_y = df_copy[label].values.reshape(df_copy[label].shape[0],)
            X = target_encoder.fit_transform(df_copy[features], reshaped_y)
            clf = CatBoostClassifier(verbose=False)
            clf.fit(X, reshaped_y)
            _, noisy_labels = BCNoise(clf, noise_level=s).simulate_noise(X, reshaped_y)
        else:        
            lcm = np.array([[1-r,r],[s,1-s]])
            noisy_labels = flip_labels_cc(df_copy.label,lcm)

        idx = (df_copy.label != noisy_labels)
        df_copy.loc[idx,'noise'] = 1
        df_copy['label'] = noisy_labels
    
    return df_copy


def train_valid_split(df, split=0.7, shuffle=True, sort_key='creation_date'):
    if shuffle:
        df = df.sample(frac=1).reset_index(drop=True)
    else:
        df = df.sort_values(by=sort_key, ignore_index=True)
    train_idx = int(round(split*df.shape[0]))
    train = df[:train_idx].reset_index(drop=True)
    valid = df[train_idx:].reset_index(drop=True)
    
    return train, valid
    

def prepare_noisy_dataset(key, noise_type, noise_amount, split=0.7, shuffle=True, 
                          sort_key='creation_date', target_encoding=False):
    """
    this function can be used to fetch datasets from FDB, 
    starts by calling prepare_data_fdb and then adding noise
    
    input: 
        key - name of FDB dataset
        noise_type - what type of noise to add
        noise_amount - how much noise to add
        split - training/validation split
        shuffle - whether or not to shuffle or sort before doing train/valid split
        sort_key - key to use to sort for train/valid split as well as weight for time-dependent noise
    """
    
    # start by getting clean dataset
    
    df, features, cat_features, label, record_id = prepare_data_fdb(key)

    if noise_type == 'boundary-consistent':
        train_and_valid = add_noise(df[df.dataset == 'train'], noise_type, noise_amount, 
                                    time_index=sort_key, features=features, cat_features=cat_features, label=label)
    else:
        train_and_valid = add_noise(df[df.dataset == 'train'], noise_type, noise_amount, time_index=sort_key)
        
    train, valid = train_valid_split(train_and_valid, split, shuffle=shuffle, sort_key=sort_key)
    test = df[df.dataset == 'test'].reset_index(drop=True)
    
    train = train[features + ['noise'] + label]
    valid = valid[features + ['noise'] + label]
    test = test[features + ['noise'] + label]
    
    if target_encoding:
        warnings.filterwarnings("ignore", category=FutureWarning)
        target_encoder = TargetEncoder(cols=cat_features)
        reshaped_y = train[label].values.reshape(train[label].shape[0],)
        train.loc[:, features] = target_encoder.fit_transform(train[features], reshaped_y)
        valid.loc[:, features] = target_encoder.transform(valid[features])
        test.loc[:, features] = target_encoder.transform(test[features])
        cat_features = None
    
    dataset = {
        'description': f"{key} dataset with noise type: {noise_type}, noise amount: {noise_amount} ",
        'features':features,
        'cat_features':cat_features,
        'label':label,
        'record_id':record_id,
        'train':train,
        'valid':valid,
        'test':test, 
        'noise':(noise_rate(train), noise_rate(valid), noise_rate(test)),
        'fraud_level':(actual_fraud_rate(train), actual_fraud_rate(valid), actual_fraud_rate(test)),
        'observed_fraud_level':(observed_fraud_rate(train),observed_fraud_rate(valid),observed_fraud_rate(test)),
        'type_1_noise_rate':(type_1_noise_rate(train),type_1_noise_rate(valid),type_1_noise_rate(test)),
        'type_2_noise_rate':(type_2_noise_rate(train),type_2_noise_rate(valid),type_2_noise_rate(test))
    }
        
    return dataset


def dataset_stats(dataset):
    noise = dataset['noise']
    fraud_level = dataset['fraud_level']
    observed_fraud_level = dataset['observed_fraud_level']
    type_1_noise_rate = dataset['type_1_noise_rate']
    type_2_noise_rate = dataset['type_2_noise_rate']
    stats = list(zip(['train','valid','test'],noise,type_1_noise_rate,type_2_noise_rate,fraud_level,observed_fraud_level))
    print(dataset['description'])
    for stat in stats:
        print('{} - total noise rate: {:.3f}, type 1 noise rate: {:.3f}, type 2 noise rate: {:.3f},\n'
                '(actual) fraud rate: {:.3f}, observed fraud rate: {:.3f}'.format(*stat))
        
