from fdb.preprocessing import *


def load_data(key):
    if key == 'fakejob':
        obj = FakejobPreProcessor(
                key = key,
                train_percentage = 0.8,
                timestamp_col = None, 
                label_col = 'fraudulent', 
                event_id_col = 'job_id'
                )
    
    elif key == 'vechicleloan':
        obj = VehicleloanPreProcessor(
            key = key,
            train_percentage = 0.8,
            timestamp_col = None, 
            label_col = 'loan_default', 
            event_id_col = 'uniqueid',
            features_to_drop = ['disbursal_date']
            )

    elif key == 'malurl':
        obj = MalurlPreProcessor(
            key = key,
            train_percentage = 0.9,
            timestamp_col = None,
            label_col = 'type',
            event_id_col = None
        )

    elif key == 'ieee':
        obj = IEEEPreProcessor(
            key = key,
            train_percentage = 0.95,
            timestamp_col = 'transactiondt',
            label_col = 'isfraud',
            event_id_col = None,
            entity_id_col = None  # manually created in code
        )

    elif key == 'ccfraud':
        obj = CCFraudPreProcessor(
            key = key,
            train_percentage = 0.8,
            timestamp_col = 'time',
            label_col = 'class',
            event_id_col = None
        )

    elif key == 'fraudecom':
        obj = FraudecomPreProcessor(
            key = key,
            train_percentage = 0.8,
            timestamp_col = 'purchase_time',
            signup_time_col = 'signup_time',
            label_col = 'class',
            event_id_col = 'user_id',
            entity_id_col = 'device_id',
            ip_address_col = 'ip_address',
            features_to_drop = ['signup_time', 'sex']
        )

    elif key == 'sparknov':
        obj = SparknovPreProcessor(
            key = key,
            timestamp_col = 'trans_date_trans_time',
            label_col = 'is_fraud',
            event_id_col = 'trans_num',
            entity_id_col = 'merchant',
            features_to_drop = ['unix_time', 'unnamed: 0']
            )

    elif key == 'twitterbot':
        obj = TwitterbotPreProcessor(
            key = key,
            train_percentage = 0.8,
            timestamp_col = None,
            label_col = 'account_type',
            event_id_col = 'id'
        )

    elif key == 'ipblock':
        obj = IPBlocklistPreProcessor(
            key = 'ipblock',
            label_col = 'is_ip_malign'
        )

    else:
        raise ValueError('Invalid key')

    return obj