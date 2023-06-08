import logging
import pandas as pd
import numpy as np


class MicroModelError(Exception):
    """
    basic exception type for micro-model specific errors
    """
    def __init__(self, error_message):
        logging.error(error_message)


class MicroModel:
    """
    Basic wrapper for the model to be used in ensemble noise removal, ModelClass can be anything that implements
    fit and predict_proba. Mainly used by MicroModelEnsemble, user is probably not calling this directly
    """

    def __init__(self, ModelClass, *args, **kwargs):
        """
        initialization of the class, ModelClass should be a *class* not an object
        e.g. CatBoostClassifier, not CatBoostClassifier()
        """
        self.clf = ModelClass(*args, **kwargs)
        self.thresh = None

    def set_thresh(self, thresh):
        # can set a threshold to be used in model predictions
        self.thresh = thresh

    def fit(self, x, y, *args, **kwargs):
        # pass-through method to call model.fit()
        self.clf.fit(x, y.values.ravel(), *args, **kwargs)

    def predict_proba(self, x, *args, **kwargs):
        # pass-through method to call model.predict_proba()
        if 'predict_proba' in dir(self.clf):
            return self.clf.predict_proba(x, *args, **kwargs)
        else:
            raise (MicroModelError('ModelClass must implement predict_proba'))

    def predict(self, x):
        # make predictions, using either defined threshold (if set) or default value of 0.5
        if self.thresh is not None:
            t = self.thresh
        else:
            t = 0.5
        scores = self.predict_proba(x)[:, 1]
        preds = [int(s > t) for s in scores]
        return scores, preds


class MicroModelEnsemble:
    """
    Ensemble of micro-models used to remove noise
    """

    def __init__(self, ModelClass, num_clfs=16, score_type='preds_avg', *args, **kwargs):
        """
        initialization of the class, ModelClass should be a *class* not an object
        e.g. CatBoostClassifier, not CatBoostClassifier()
        params:
        ModelClass - base class to use, needs to implement fit and predict_proba
        num_clfs - number of classifiers to use in cleaning ensemble
        score_type - means of computing anomaly score from micro-model scores
        args/kwargs - any other parameters to pass to model constructor, e.g. cat_features or iterations for CatBoost
        """
        self.score_type = score_type
        
        if type(num_clfs) is not int or num_clfs <= 0:
            raise (MicroModelError('num_clfs must be a positive integer'))
        self.ModelClass = ModelClass

        # one classifier that will be trained over entire dataset
        self.big_clf = MicroModel(ModelClass=ModelClass, *args, **kwargs)

        # micro-models to later be trained over slices
        self.num_clfs = num_clfs
        self.clfs = []
        for i in range(num_clfs):
            self.clfs.append(MicroModel(ModelClass=ModelClass, *args, **kwargs))
        self.thresholds = {}

    def fit(self, x, y, *args, **kwargs):
        # assumption that data is already shuffled or sorted (by date or other appropriate key)
        # according to the usecase

        if not isinstance(y, pd.DataFrame):
            y = pd.DataFrame(y)

            # fit one classifier on all the data
        self.big_clf.fit(x, y, *args, **kwargs)

        # now fit individual models on slices of data
        stride = round(x.shape[0] / self.num_clfs)
        for i, clf in enumerate(self.clfs):
            idx = slice(i * stride, min((i + 1) * stride, x.shape[0]))
            x_i = x.iloc[idx, :]
            y_i = y.iloc[idx, :]
            clf.fit(x_i, y_i, *args, **kwargs)

    def predict_proba(self, x, *args, **kwargs):
        # output is the mean of the (binary) predictions of all models in the ensemble
        # e.g. the percentage of models that voted on the example
        results = pd.DataFrame(index=np.arange(x.shape[0]))
        if self.score_type == 'preds_avg':
            for i, clf in enumerate(self.clfs):
                _, results[i] = clf.predict(x, *args, **kwargs)
        elif self.score_type == 'score_avg':
            for i, clf in enumerate(self.clfs):
                results[i] = clf.predict_proba(x, *args, **kwargs)[:, 1]

        scores = results.mean(axis=1, numeric_only=True)
        return scores

    def predict(self, x, threshold=0.5, *args, **kwargs):
        # compare output of predict_proba to a threshold in order to make a binary prediction, default is 0.5
        scores = self.predict_proba(x)
        preds = np.array([int(s >= threshold) for s in scores])
        return scores, preds

    def filter_noise(self, x, y, pulearning=True, threshold=0.5):
        # compare ensemble predictions to observed labels and return the examples that are NOT considered noise
        # i.e. this is noise REMOVAL
        # pu_learning=True means a class-conditional assumption is being made,
        # there no examples of true 0s mislabeled as 1s
        scores, susp = self.predict(x, threshold)
        if pulearning:
            conf = ((y == 1) | ((y == 0) & (susp == 0)))
        else:
            conf = (((y == 1) & (scores > 1 - threshold)) | ((y == 0) & (scores < threshold)))

        return x[conf].reset_index(drop=True), y[conf]

    def clean_noise(self, x, y, pulearning=True, threshold=0.5):
        # compare ensemble predictions to observed labels and return all examples with corrected labels
        # i.e. this is noise CLEANING
        # pu_learning=True means a class-conditional assumption is being made,
        # there no examples of true 0s mislabeled as 1s
        x = x.copy()
        y = y.copy()
        _, susp = self.predict(x, threshold)
        # flip all the probable 1s to actual 1s
        probable_1 = (y == 0) & (susp == 1)
        y[probable_1] = 1
        if not pulearning:
            # if there are both types of noise, flip probable 0s to actual 0s
            probable_0 = (y == 1) & (susp == 0)
            y[probable_0] = 0

        return x, y


class MicroModelCleaner:
    """
    This class performs the entire model training process end-to-end - given a dataset it will first train an ensemble
    then remove noise, then train a final model on the clean data
    """

    def __init__(self, ModelClass, strategy='filter', pulearning=True, num_clfs=16, threshold=0.5, *args, **kwargs):
        """
        initialization of the class, ModelClass should be a *class* not an object
        e.g. CatBoostClassifier, not CatBoostClassifier()
        params:
        ModelClass - base class to use, needs to implement fit and predict_proba
        strategy - whether to remove noise ('filter') or flip labels ('clean')
        pulearning - class-conditional assumption, if True assume there is no true 0's mislabeled as 1's
        num_clfs - number of classifiers to use in cleaning ensemble
        threshold - percentage of classifiers that have to vote to remove noise (0.5 is majority voting)
        args/kwargs - any other parameters to pass to model constructor, e.g. cat_features or iterations for CatBoost
        """
        self.detector = MicroModelEnsemble(ModelClass, num_clfs, *args, **kwargs)
        self.clf = ModelClass(*args, **kwargs)
        if strategy.lower() not in ['filter', 'clean']:
            raise (MicroModelError('strategy must be filter or clean'))
        self.strategy = strategy.lower()
        self.pulearning = pulearning
        self.threshold = threshold

    def fit(self, x, y, *args, **kwargs):
        # first train the Ensemble to deal with the noise
        self.detector.fit(x, y, *args, **kwargs)
        if self.strategy == 'filter':
            x_clean, y_clean = self.detector.filter_noise(x, y, self.pulearning, self.threshold)
        else:
            x_clean, y_clean = self.detector.clean_noise(x, y, self.pulearning, self.threshold)

        # then train final model on clean data
        self.clf.fit(x_clean, y_clean, *args, **kwargs)

    def predict(self, x, *args, **kwargs):
        return self.clf.predict(x, *args, **kwargs)

    def predict_proba(self, x, *args, **kwargs):
        return self.clf.predict_proba(x, *args, **kwargs)

