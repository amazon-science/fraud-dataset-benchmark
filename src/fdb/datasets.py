from abc import abstractmethod, ABC
from fdb.preprocessing import *
from fdb.preprocessing_objects import load_data
from sklearn.metrics import roc_auc_score, roc_curve, auc

class FraudDatasetBenchmark(ABC):
    def __init__(
        self, 
        key, 
        load_pre_downloaded=False,
        delete_downloaded=True,
        add_random_values_if_real_na = {
            "EVENT_TIMESTAMP": True,
            "LABEL_TIMESTAMP": True,
            "ENTITY_ID": True,
            "ENTITY_TYPE": True,
            "EVENT_ID": True
            }):
        self.key = key
        self.obj = load_data(self.key, load_pre_downloaded, delete_downloaded, add_random_values_if_real_na)
    
    @property
    def train(self):
        return self.obj.train

    @property
    def test(self):
        return self.obj.test

    @property
    def test_labels(self):
        return self.obj.test_labels

    def eval(self, y_pred):
        
        """
        Method to evaluate predictions against the test set
        """
        roc_score = roc_auc_score(self.test_labels['EVENT_LABEL'], y_pred)
        fpr, tpr, thres = roc_curve(self.test_labels['EVENT_LABEL'], y_pred)
        tpr_1fpr = np.interp(0.01, fpr, tpr)
        metrics = {'roc_score': roc_score, 'tpr_1fpr': tpr_1fpr}
        return metrics


