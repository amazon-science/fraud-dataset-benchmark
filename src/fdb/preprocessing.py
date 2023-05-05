

import os
import re
import shutil
import kaggle
import pkgutil
import requests
import zipfile
import numpy as np
from abc import ABC
import pandas as pd
import socket, struct
from faker import Faker
from zipfile import ZipFile
from datetime import datetime
from datetime import timedelta
from io import StringIO, BytesIO
from dateutil.relativedelta import relativedelta

from fdb.kaggle_configs import KAGGLE_CONFIGS

fake = Faker(['en_US'])


# Naming convention for the meta data columns in standardized datasets
_EVENT_TIMESTAMP = 'EVENT_TIMESTAMP'  # timestamp column
_ENTITY_TYPE = 'ENTITY_TYPE'  # afd specific requirement
_EVENT_LABEL = 'EVENT_LABEL'  # label column
_EVENT_ID = 'EVENT_ID'  # transaction/event id 
_ENTITY_ID = 'ENTITY_ID'  # represents user/account id
_LABEL_TIMESTAMP = 'LABEL_TIMESTAMP'  # added in a cases where entity id is meaninful

# Kaggle config related strings
_OWNER = 'owner'
_COMPETITIONS = 'competitions'
_TYPE = 'type'
_FILENAME = 'filename'
_DATASETS = 'datasets'
_DATASET = 'dataset'
_VERSION = 'version'

# Some fixed parameters
_RANDOM_STATE = 1
_CWD = os.getcwd()
_DOWNLOAD_LOCATION = os.path.join(_CWD, 'tmp')
_TIMESTAMP_FORMAT = '%Y-%m-%dT%H:%M:%SZ'
_DEFAULT_LABEL_TIMESTAMP = datetime.now().strftime(_TIMESTAMP_FORMAT)


class BasePreProcessor(ABC):
    def __init__(
        self, 
        key = None, 
        train_percentage = 0.8,
        timestamp_col = None, 
        label_col = None, 
        label_timestamp_col = None,
        event_id_col = None,
        entity_id_col = None,
        features_to_drop = [],
        load_pre_downloaded = False,
        delete_downloaded = True,
        add_random_values_if_real_na = {
            "EVENT_TIMESTAMP": True,
            "LABEL_TIMESTAMP": True,
            "ENTITY_ID": True,
            "ENTITY_TYPE": True,
            "EVENT_ID": True
            }
        ):
        
        self.key = key 
        self.train_percentage = train_percentage
        self.features_to_drop = features_to_drop
        self.delete_downloaded = delete_downloaded
        
        self._timestamp_col = timestamp_col
        self._label_col = label_col
        self._label_timestamp_col = label_timestamp_col
        self._event_id_col = event_id_col
        self._entity_id_col = entity_id_col
        self._add_random_values_if_real_na = add_random_values_if_real_na
        
        # Simply get all required objects at the time of object creation
        if KAGGLE_CONFIGS.get(self.key) and not load_pre_downloaded:
            self.download_kaggle_data()  # download the data when an object is created
        self.load_data()
        self.preprocess()
        self.train_test_split()


    def _download_kaggle_data_from_competetions(self):
        file_name = KAGGLE_CONFIGS[self.key][_OWNER]
        kaggle.api.competition_download_files(
            competition = KAGGLE_CONFIGS[self.key][_OWNER],
            path = _DOWNLOAD_LOCATION
        )
        return file_name

    def _download_kaggle_data_from_datasets_with_given_filename(self):
        file_name = KAGGLE_CONFIGS[self.key][_FILENAME]
        response = kaggle.api.datasets_download_file(
            owner_slug = KAGGLE_CONFIGS[self.key][_OWNER],
            dataset_slug = KAGGLE_CONFIGS[self.key][_DATASET],
            file_name = file_name,
            dataset_version_number=KAGGLE_CONFIGS[self.key][_VERSION],
            _preload_content = False,
        )
        with open(os.path.join(_DOWNLOAD_LOCATION, file_name + '.zip'), 'wb') as f:
            f.write(response.data)
        return file_name

    def _download_kaggle_data_from_datasets_containing_single_file(self):
        file_name = KAGGLE_CONFIGS[self.key][_DATASET]
        kaggle.api.dataset_download_files(
            dataset = os.path.join(KAGGLE_CONFIGS[self.key][_OWNER], KAGGLE_CONFIGS[self.key][_DATASET]),
            path = _DOWNLOAD_LOCATION
        )
        return file_name

    def download_kaggle_data(self):
        """
        Download and extract the data from Kaggle. Puts the data in tmp directory within current directory.
        """

        if not os.path.exists(_DOWNLOAD_LOCATION):
            os.mkdir(_DOWNLOAD_LOCATION)

        print('Data download location', _DOWNLOAD_LOCATION)
            
        
        if KAGGLE_CONFIGS[self.key][_TYPE] == _COMPETITIONS:
            file_name = self._download_kaggle_data_from_competetions()
                 
        elif KAGGLE_CONFIGS[self.key][_TYPE] == _DATASETS:
            # If filename is given, download single file,
            # Else download all files.
            if KAGGLE_CONFIGS[self.key].get(_FILENAME):
                file_name = self._download_kaggle_data_from_datasets_with_given_filename()
            else:
                file_name = self._download_kaggle_data_from_datasets_containing_single_file()
                
        else:
            raise ValueError('Type should be among competetions or datasets in config')
        
        with zipfile.ZipFile(os.path.join(_DOWNLOAD_LOCATION, file_name + '.zip'), 'r') as zip_ref:
            zip_ref.extractall(_DOWNLOAD_LOCATION)

    def load_data(self):
        self.df = pd.read_csv(os.path.join(_DOWNLOAD_LOCATION, KAGGLE_CONFIGS[self.key]['filename']), dtype='object')
        # delete downloaded data after loading in memory
        if self.delete_downloaded: shutil.rmtree(_DOWNLOAD_LOCATION)

    @property
    def timestamp_col(self):
        return self._timestamp_col  # If timestamp not available, will create fake timestamps

    @property
    def label_col(self):
        if self._label_col is None:
            raise ValueError('Label column not specified')
        else:
            return self._label_col

    @property
    def event_id_col(self):
        return self._event_id_col  # If event id not available, will create fake event ids
    
    @property
    def entity_id_col(self):
        return self._entity_id_col

    def standardize_timestamp_col(self):
        if self.timestamp_col is not None:
            self.df[_EVENT_TIMESTAMP] = pd.to_datetime(self.df[self.timestamp_col]).apply(lambda x: x.strftime(_TIMESTAMP_FORMAT))
            self.df.drop(self.timestamp_col, axis=1, inplace=True)
        elif self.timestamp_col is None and self._add_random_values_if_real_na[_EVENT_TIMESTAMP]:
            self.df[_EVENT_TIMESTAMP] = self.df[_EVENT_LABEL].apply(
                lambda x: fake.date_time_between(
                    start_date='-1y',   # think about making it to fixed date. vs from now?
                    end_date='now',
                    tzinfo=None).strftime(_TIMESTAMP_FORMAT))
        
        if self._label_timestamp_col is None and self._add_random_values_if_real_na[_LABEL_TIMESTAMP]:
            self.df[_LABEL_TIMESTAMP] = _DEFAULT_LABEL_TIMESTAMP # most recent date 
        elif self._label_timestamp_col is not None:
            self.df[_LABEL_TIMESTAMP] = pd.to_datetime(self.df[self._label_timestamp_col]).apply(lambda x: x.strftime(_TIMESTAMP_FORMAT))
            self.df.drop(self._label_timestamp_col, axis=1, inplace=True)

    def standardize_label_col(self):
        self.df.rename({self.label_col: _EVENT_LABEL}, axis=1, inplace=True)
        self.df[_EVENT_LABEL] = self.df[_EVENT_LABEL].astype(int)

    def standardize_event_id_col(self):
        if self.event_id_col is not None:
            self.df.rename({self.event_id_col: _EVENT_ID}, axis=1, inplace=True)
            self.df[_EVENT_ID] = self.df[_EVENT_ID].astype(str)
        elif self.event_id_col is None and self._add_random_values_if_real_na[_EVENT_ID]: # add fake one if not exist
            self.df[_EVENT_ID] = self.df[_EVENT_LABEL].apply(
                lambda x: fake.uuid4())

            
    def standardize_entity_id_col(self):
        if self.entity_id_col is not None:
            self.df.rename({self.entity_id_col: _ENTITY_ID}, axis=1, inplace=True)
        elif self.entity_id_col is None and self._add_random_values_if_real_na[_ENTITY_ID]: # add fake one if not exist
            self.df[_ENTITY_ID] = self.df[_EVENT_LABEL].apply(
                lambda x: fake.uuid4())

    def rename_features(self):
        rename_map = {} # default is empty map that won't rename any columns
        self.df.rename(rename_map, axis=1, inplace=True)

    def subset_features(self):
        features_to_select = self.df.columns.tolist()
        self.df = self.df[features_to_select]  # all by default
    
    def drop_features(self):
        self.df.drop(self.features_to_drop, axis=1, inplace=True)

    def add_meta_data(self):
        if self._add_random_values_if_real_na[_ENTITY_TYPE]: 
            self.df[_ENTITY_TYPE] = 'user'

    def sort_by_timestamp(self):
        self.df.sort_values(by=_EVENT_TIMESTAMP, ascending=True, inplace=True)

    def lower_case_col_names(self):
         self.df.columns = [s.lower() for s in self.df.columns]
        
    def preprocess(self):
        self.lower_case_col_names()
        self.standardize_label_col()
        self.standardize_event_id_col()
        self.standardize_entity_id_col()
        self.standardize_timestamp_col()
        self.add_meta_data()
        self.rename_features()
        self.subset_features()
        self.drop_features()
        if self.timestamp_col:
            self.sort_by_timestamp()

    def train_test_split(self):
        """
        Default setting is out of time with 80%-20% into training and testing respectively
        """
        if self.timestamp_col: 
            split_pt = int(self.df.shape[0]*self.train_percentage)
            self.train = self.df.copy().iloc[:split_pt, :]
            self.test = self.df.copy().iloc[split_pt:, :]
        else:  # random if no timestamp col available
            self.train = self.df.sample(frac=self.train_percentage, random_state=_RANDOM_STATE)
            self.test = self.df.copy()[~self.df.index.isin(self.train.index)]
            self.test.reset_index(drop=True, inplace=True)
        
        self.test_labels = self.test[[_EVENT_LABEL]]
        if self.event_id_col is None and self._add_random_values_if_real_na[_EVENT_ID]:
            self.test_labels[_EVENT_ID] = self.test[_EVENT_ID]
        self.test.drop([_EVENT_LABEL, _LABEL_TIMESTAMP], axis=1, inplace=True, errors="ignore")


class FakejobPreProcessor(BasePreProcessor):
    def __init__(self, **kw):
        super(FakejobPreProcessor, self).__init__(**kw)


class VehicleloanPreProcessor(BasePreProcessor):
    def __init__(self, **kw):
        super(VehicleloanPreProcessor, self).__init__(**kw)


class MalurlPreProcessor(BasePreProcessor):
    """
    This one originally multiple classes for manignant. 
    We will combine all malignant one class to keep benchmark binary for now
    
    """
    def __init__(self, **kw):
        super(MalurlPreProcessor, self).__init__(**kw)

    def standardize_label_col(self):
        self.df.rename({self.label_col: _EVENT_LABEL}, axis=1, inplace=True)
        binary_mapper = {
            'defacement': 1,
            'phishing': 1,
            'malware': 1,
            'benign': 0
        }
        
        self.df[_EVENT_LABEL] = self.df[_EVENT_LABEL].map(binary_mapper)

    def add_dummy_col(self):
        self.df['dummy_cat'] = self.df[_EVENT_LABEL].apply(lambda x: fake.uuid4())

    def preprocess(self):
        super(MalurlPreProcessor, self).preprocess()
        self.add_dummy_col()

class IEEEPreProcessor(BasePreProcessor):
    """
    Some pre-processing was done using kaggle kernels below.  

    References:
        Data Source: https://www.kaggle.com/c/ieee-fraud-detection/data

        Some processing from: https://www.kaggle.com/cdeotte/xgb-fraud-with-magic-0-9600
        Feature selection to reduce to 100: https://www.kaggle.com/code/pavelvpster/ieee-fraud-feature-selection-rfecv/notebook

    """
    def __init__(self, **kw):
        super(IEEEPreProcessor, self).__init__(**kw)

    @staticmethod
    def _dtypes_cols():

        # FIRST 53 COLUMNS
        cols = ['TransactionID', 'TransactionDT', 'TransactionAmt',
            'ProductCD', 'card1', 'card2', 'card3', 'card4', 'card5', 'card6',
            'addr1', 'addr2', 'dist1', 'dist2', 'P_emaildomain', 'R_emaildomain',
            'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11',
            'C12', 'C13', 'C14', 'D1', 'D2', 'D3', 'D4', 'D5', 'D6', 'D7', 'D8',
            'D9', 'D10', 'D11', 'D12', 'D13', 'D14', 'D15', 'M1', 'M2', 'M3', 'M4',
            'M5', 'M6', 'M7', 'M8', 'M9']

        # V COLUMNS TO LOAD DECIDED BY CORRELATION EDA
        # https://www.kaggle.com/cdeotte/eda-for-columns-v-and-id
        v =  [1, 3, 4, 6, 8, 11]
        v += [13, 14, 17, 20, 23, 26, 27, 30]
        v += [36, 37, 40, 41, 44, 47, 48]
        v += [54, 56, 59, 62, 65, 67, 68, 70]
        v += [76, 78, 80, 82, 86, 88, 89, 91]

        #v += [96, 98, 99, 104] #relates to groups, no NAN 
        v += [107, 108, 111, 115, 117, 120, 121, 123] # maybe group, no NAN
        v += [124, 127, 129, 130, 136] # relates to groups, no NAN

        # LOTS OF NAN BELOW
        v += [138, 139, 142, 147, 156, 162] #b1
        v += [165, 160, 166] #b1
        v += [178, 176, 173, 182] #b2
        v += [187, 203, 205, 207, 215] #b2
        v += [169, 171, 175, 180, 185, 188, 198, 210, 209] #b2
        v += [218, 223, 224, 226, 228, 229, 235] #b3
        v += [240, 258, 257, 253, 252, 260, 261] #b3
        v += [264, 266, 267, 274, 277] #b3
        v += [220, 221, 234, 238, 250, 271] #b3

        v += [294, 284, 285, 286, 291, 297] # relates to grous, no NAN
        v += [303, 305, 307, 309, 310, 320] # relates to groups, no NAN
        v += [281, 283, 289, 296, 301, 314] # relates to groups, no NAN

        # COLUMNS WITH STRINGS
        str_type = ['ProductCD', 'card4', 'card6', 'P_emaildomain', 'R_emaildomain','M1', 'M2', 'M3', 'M4','M5',
                    'M6', 'M7', 'M8', 'M9', 'id_12', 'id_15', 'id_16', 'id_23', 'id_27', 'id_28', 'id_29', 'id_30', 
                    'id_31', 'id_33', 'id_34', 'id_35', 'id_36', 'id_37', 'id_38', 'DeviceType', 'DeviceInfo']
        str_type += ['id-12', 'id-15', 'id-16', 'id-23', 'id-27', 'id-28', 'id-29', 'id-30', 
            'id-31', 'id-33', 'id-34', 'id-35', 'id-36', 'id-37', 'id-38']


        cols += ['V'+str(x) for x in v]
        dtypes = {}
        for c in cols+['id_0'+str(x) for x in range(1,10)]+['id_'+str(x) for x in range(10,34)]+\
            ['id-0'+str(x) for x in range(1,10)]+['id-'+str(x) for x in range(10,34)]:
                dtypes[c] = 'float32'
        for c in str_type: dtypes[c] = 'category'

        return dtypes, cols


    def load_data(self):
        """
        Hard coded file names for this dataset as it contains multiple files to be combined
        """

        dtypes, cols = IEEEPreProcessor._dtypes_cols()

        self.df = pd.read_csv(
            os.path.join(_DOWNLOAD_LOCATION,
             'train_transaction.csv'), 
             index_col='TransactionID',
             dtype=dtypes, 
             usecols=cols+['isFraud'])

        self.df_id = pd.read_csv(
            os.path.join(_DOWNLOAD_LOCATION, 
            'train_identity.csv'),
            index_col='TransactionID', 
            dtype=dtypes)
        self.df = self.df.merge(self.df_id, how='left', left_index=True, right_index=True)

        # delete downloaded data after loading in memory
        if self.delete_downloaded: shutil.rmtree(_DOWNLOAD_LOCATION)

    def normalization(self):
        # NORMALIZE D COLUMNS
        for i in range(1,16):
            if i in [1,2,3,5,9]: continue
            self.df['d'+str(i)] =  self.df['d'+str(i)] - self.df[self.timestamp_col]/np.float32(24*60*60)

    def standardize_entity_id_col(self):
        def _encode_CB(col1, col2, df):
            nm = col1+'_'+col2
            df[nm] = df[col1].astype(str)+'_'+df[col2].astype(str)
        
        _encode_CB('card1', 'addr1', self.df)
        self.df['day'] = self.df[self.timestamp_col] / (24*60*60)
        self.df[_ENTITY_ID] = self.df['card1_addr1'].astype(str) + '_' + np.floor(self.df['day'] - self.df['d1']).astype(str)

    @staticmethod
    def _add_seconds(x):
        init_time = '2021-01-01T00:00:00Z'
        dt_format = _TIMESTAMP_FORMAT
        init_time = datetime.strptime(init_time, dt_format) # start date from last 18 months
        final_time = init_time + timedelta(seconds=x)
        return final_time.strftime(_TIMESTAMP_FORMAT)   

    def standardize_timestamp_col(self):
        self.df[_EVENT_TIMESTAMP] = self.df[self.timestamp_col].apply(lambda x: IEEEPreProcessor._add_seconds(x))        
        self.df.drop(self.timestamp_col, axis=1, inplace=True)
        if self._add_random_values_if_real_na["LABEL_TIMESTAMP"]:
            self.df[_LABEL_TIMESTAMP] = _DEFAULT_LABEL_TIMESTAMP # most recent date 

    def subset_features(self):
        features_to_select = \
         ['transactionamt', 'productcd', 'card1', 'card2', 'card3', 'card5', 'card6', 'addr1', 'dist1',
         'p_emaildomain', 'r_emaildomain', 'c1', 'c2', 'c4', 'c5', 'c6', 'c7', 'c8', 'c9', 'c10', 'c11',
         'c12', 'c13', 'c14', 'v62', 'v70', 'v76', 'v78', 'v82', 'v91', 'v127', 'v130', 'v139', 'v160',
         'v165', 'v187', 'v203', 'v207', 'v209', 'v210', 'v221', 'v234', 'v257', 'v258', 'v261', 'v264',
         'v266', 'v267', 'v271', 'v274', 'v277', 'v283', 'v285', 'v289', 'v291', 'v294', 'id_01', 'id_02',
         'id_05', 'id_06', 'id_09', 'id_13', 'id_17', 'id_19', 'id_20', 'devicetype', 'deviceinfo',
         'EVENT_TIMESTAMP', 'ENTITY_ID', 'ENTITY_TYPE', 'EVENT_ID', 'EVENT_LABEL', 'LABEL_TIMESTAMP']
        self.df = self.df.loc[:, self.df.columns.isin(features_to_select)]

    def preprocess(self):
        self.lower_case_col_names()
        self.normalization()  # normalize D columns
        self.standardize_label_col()
        self.standardize_event_id_col()
        self.standardize_entity_id_col()
        self.standardize_timestamp_col()
        self.add_meta_data()
        self.rename_features()
        self.subset_features()  
        if self.timestamp_col:
            self.sort_by_timestamp()


class CCFraudPreProcessor(BasePreProcessor):
    def __init__(self, **kw):
        super(CCFraudPreProcessor, self).__init__(**kw)

    @staticmethod
    def _add_minutes(x):
        dt_format = _TIMESTAMP_FORMAT
        init_time = datetime.strptime('2021-09-01T00:00:00Z', dt_format)  # chose randomly but in last 18 months
        final_time = init_time + timedelta(minutes=x)
        return final_time.strftime(_TIMESTAMP_FORMAT)   

    def standardize_timestamp_col(self):
        self.df[_EVENT_TIMESTAMP] = self.df[self.timestamp_col].astype(float).apply(lambda x: CCFraudPreProcessor._add_minutes(x))        
        self.df.drop(self.timestamp_col, axis=1, inplace=True)
        if self._add_random_values_if_real_na[_LABEL_TIMESTAMP]:
            self.df[_LABEL_TIMESTAMP] = _DEFAULT_LABEL_TIMESTAMP # most recent date 
        
class FraudecomPreProcessor(BasePreProcessor):
    def __init__(self, ip_address_col, signup_time_col, **kw):
        self.ip_address_col = ip_address_col
        self.signup_time_col = signup_time_col
        super(FraudecomPreProcessor, self).__init__(**kw)

    @staticmethod
    def _add_years(init_time):
        dt_format = '%Y-%m-%d %H:%M:%S'
        init_time = datetime.strptime(init_time, dt_format)
        final_time = init_time + relativedelta(years=6)  # move to more recent time range
        return final_time.strftime(_TIMESTAMP_FORMAT) 


    def standardize_timestamp_col(self):

        self.df[_EVENT_TIMESTAMP] = self.df[self.timestamp_col].apply(lambda x: FraudecomPreProcessor._add_years(x))        
        self.df.drop(self.timestamp_col, axis=1, inplace=True)

        # Also add _LABEL_TIMESTAMP to allow training of this dataset with TFI
        if self._add_random_values_if_real_na[_LABEL_TIMESTAMP]:
            self.df[_LABEL_TIMESTAMP] = _DEFAULT_LABEL_TIMESTAMP # most recent date 

    def process_ip(self):
        """
        This dataset has ip address as a feature, but needs to be converted into standard IPV4.
        """
        self.df[self.ip_address_col] = self.df[self.ip_address_col].astype(float).astype(int).\
                                        apply(lambda x: socket.inet_ntoa(struct.pack('!L', x)))

    def create_time_since_signup(self):
        self.df['time_since_signup'] = (
            pd.to_datetime(self.df[self.timestamp_col]) -\
            pd.to_datetime(self.df[self.signup_time_col])).dt.seconds

    def preprocess(self):
        self.lower_case_col_names()
        self.standardize_label_col()
        self.standardize_event_id_col()
        self.standardize_entity_id_col()
        self.create_time_since_signup()  # One manually engineered feature
        self.standardize_timestamp_col()
        self.add_meta_data()
        self.process_ip()  # This extra step added
        self.rename_features()
        self.drop_features()  # Replace select with drop
        if self.timestamp_col:
            self.sort_by_timestamp()


class SparknovPreProcessor(BasePreProcessor):
    def __init__(self, **kw):
        super(SparknovPreProcessor, self).__init__(**kw)
        
    def load_data(self):
        """
        Hard coded file names for this dataset as it contains multiple files to be combined
        """

        df_train = pd.read_csv(os.path.join(_DOWNLOAD_LOCATION,'fraudTrain.csv'))
        df_train['seg'] = 'train'

        df_test = pd.read_csv(os.path.join(_DOWNLOAD_LOCATION,'fraudTest.csv'))
        df_test['seg'] = 'test'

        self.df = pd.concat([df_train, df_test], ignore_index=True)

        # delete downloaded data after loading in memory
        if self.delete_downloaded: shutil.rmtree(_DOWNLOAD_LOCATION)

    @staticmethod
    def _add_months(x):
        _TIMESTAMP_FORMAT_SPARKNOV = '%Y-%m-%d %H:%M:%S'

        x = datetime.strptime(x, _TIMESTAMP_FORMAT_SPARKNOV)  
        final_time = x + relativedelta(months=20) # chosen to move dates close to now()
        return final_time.strftime(_TIMESTAMP_FORMAT)    

    def standardize_timestamp_col(self):

        self.df[_EVENT_TIMESTAMP] = self.df[self.timestamp_col].apply(lambda x: SparknovPreProcessor._add_months(x))        
        self.df.drop(self.timestamp_col, axis=1, inplace=True)
        self.df[_LABEL_TIMESTAMP] = _DEFAULT_LABEL_TIMESTAMP # most recent date 

    def standardize_entity_id_col(self):

        self.df.rename({self.entity_id_col: _ENTITY_ID}, axis=1, inplace=True)
        self.df[_ENTITY_ID] = self.df[_ENTITY_ID].\
                                str.lower().\
                                apply(lambda x: re.sub(r'[^A-Za-z0-9]+', '_', x))
        
    def train_test_split(self):
        self.train = self.df.copy()[self.df['seg'] == 'train']
        self.train.reset_index(drop=True, inplace=True)
        self.train.drop(['seg'], axis=1, inplace=True)
        
        self.test = self.df.copy()[self.df['seg'] == 'test']
        self.test.reset_index(drop=True, inplace=True)
        self.test.drop(['seg'], axis=1, inplace=True)
        self.test = self.test.sample(n=20000, random_state=1)
        
        self.test_labels = self.test[[_EVENT_LABEL]]
        if self.event_id_col is None and self._add_random_values_if_real_na[_EVENT_ID]:
            self.test_labels[_EVENT_ID] = self.test[_EVENT_ID]
        self.test.drop([_EVENT_LABEL, _LABEL_TIMESTAMP], axis=1, inplace=True, errors="ignore")


class TwitterbotPreProcessor(BasePreProcessor):
    def __init__(self, **kw):
        super(TwitterbotPreProcessor, self).__init__(**kw)

    def standardize_label_col(self):
        self.df.rename({self.label_col: _EVENT_LABEL}, axis=1, inplace=True)
        binary_mapper = {
            'bot': 1,
            'human': 0
        }
        
        self.df[_EVENT_LABEL] = self.df[_EVENT_LABEL].map(binary_mapper)


class IPBlocklistPreProcessor(BasePreProcessor):
    """
    The dataset source is http://cinsscore.com/list/ci-badguys.txt. 
    In order to download/access the latest version of this dataset, a sign-in/sign-up to is not required

    Since this dataset is not version controlled from the source, we added the version of dataset we used for experiments
    discussed in the paper. The versioned dataset is as of 2022-06-07. 
    The code is set to pick the fixed version. If the user is interested to use the latest version,
    'version' argument will need to be turned off (i.e. set to None) 
    """
    def __init__(self, version, **kw):
        self.version = version  # string or None. If string, picks one from versioned_datasets, else creates one from source  
        super(IPBlocklistPreProcessor, self).__init__(**kw)
        
    def load_data(self):
        if self.version is None:
            # load malicious IPs from the source
            _URL = 'http://cinsscore.com/list/ci-badguys.txt'  # contains confirmed malicious IPs
            _N_BENIGN = 200000
            
            res = requests.get(_URL)
            ip_mal = pd.read_csv(StringIO(res.text), sep='\n', names=['ip'], header=None)
            ip_mal['is_ip_malign'] = 1
            
            # add fake IPs as benign
            ip_ben = pd.DataFrame({
                'ip': [fake.ipv4() for i in range(_N_BENIGN)], 
                'is_ip_malign': 0
            })
            
            self.df = pd.concat([ip_mal, ip_ben], axis=0, ignore_index=True)
        else:

            _VERSIONED_DATA_PATH = f'versioned_datasets/{self.key}/{self.version}.zip'
            data = pkgutil.get_data(__name__, _VERSIONED_DATA_PATH)
            with zipfile.ZipFile(BytesIO(data)) as f:
                self.train = pd.read_csv(f.open('train.csv'))
                self.test = pd.read_csv(f.open('test.csv'))
                self.test_labels = pd.read_csv(f.open('test_labels.csv'))

    def add_dummy_col(self):
        self.df['dummy_cat'] = self.df[_EVENT_LABEL].apply(lambda x: fake.uuid4())
    
    def train_test_split(self):
        if self.version is None:
            super(IPBlocklistPreProcessor, self).train_test_split()
        
    def preprocess(self):
        if self.version is None:
            super(IPBlocklistPreProcessor, self).preprocess()
            self.add_dummy_col()      
