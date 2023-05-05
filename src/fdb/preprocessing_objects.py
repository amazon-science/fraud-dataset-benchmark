from fdb.preprocessing import *


def load_data(key, load_pre_downloaded, delete_downloaded, add_random_values_if_real_na):
    common_kw = {
        "key": key,
        "load_pre_downloaded": load_pre_downloaded,
        "delete_downloaded": delete_downloaded,
        "add_random_values_if_real_na": add_random_values_if_real_na
    }

    if key == 'fakejob':
        obj = FakejobPreProcessor(
                train_percentage = 0.8,
                timestamp_col = None, 
                label_col = 'fraudulent', 
                event_id_col = 'job_id',
                **common_kw
                )
    
    elif key == 'vehicleloan':
        obj = VehicleloanPreProcessor(
            train_percentage = 0.8,
            timestamp_col = None, 
            label_col = 'loan_default', 
            event_id_col = 'uniqueid',
            features_to_drop = ['disbursal_date'],
            **common_kw
            )

    elif key == 'malurl':
        obj = MalurlPreProcessor(
            train_percentage = 0.9,
            timestamp_col = None,
            label_col = 'type',
            event_id_col = None,
            **common_kw
        )

    elif key == 'ieeecis':
        obj = IEEEPreProcessor(
            train_percentage = 0.95,
            timestamp_col = 'transactiondt',
            label_col = 'isfraud',
            event_id_col = None,
            entity_id_col = None,  # manually created in code
            **common_kw
        )

    elif key == 'ccfraud':
        obj = CCFraudPreProcessor(
            train_percentage = 0.8,
            timestamp_col = 'time',
            label_col = 'class',
            event_id_col = None,
            **common_kw
        )

    elif key == 'fraudecom':
        obj = FraudecomPreProcessor(
            train_percentage = 0.8,
            timestamp_col = 'purchase_time',
            signup_time_col = 'signup_time',
            label_col = 'class',
            event_id_col = 'user_id',
            entity_id_col = 'device_id',
            ip_address_col = 'ip_address',
            features_to_drop = ['signup_time', 'sex'],
            **common_kw
        )

    elif key == 'sparknov':
        obj = SparknovPreProcessor(
            timestamp_col = 'trans_date_trans_time',
            label_col = 'is_fraud',
            event_id_col = 'trans_num',
            entity_id_col = 'merchant',
            features_to_drop = ['unix_time', 'unnamed: 0'],
            **common_kw
            )

    elif key == 'twitterbot':
        obj = TwitterbotPreProcessor(
            train_percentage = 0.8,
            timestamp_col = None,
            label_col = 'account_type',
            event_id_col = 'id',
            **common_kw
        )

    elif key == 'ipblock':
        obj = IPBlocklistPreProcessor(
            label_col = 'is_ip_malign',
            version = '20220607',
            **common_kw
        )

    else:
        raise ValueError('Invalid key')

    return obj