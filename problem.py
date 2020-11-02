import os
import pickle

import numpy as np
import rampwf as rw
from sklearn.model_selection import train_test_split, ShuffleSplit

problem_title = 'Optical network modelling'
_NB_CHANNELS = 32  # C100

# We are splitting both train/test and train/valid using the campaign
# indices. Training campaigns will be all subcascades, and they fully
# go in the training set. Each train/valid split on the training set
# is then using _cv_valid_rate of the training instances for training.
# .They will not be part of the validation.
# Test campaigns will be split: _test_rate of them will be in the test
# set and (1 - _test_rate) in the training set. Of this latter, 
# _cv_valid_rate will be in each fold validation set, and
# (1 - _cv_valid_rate) will be part of each fold training set.
_train_campaigns = [1, 2]
_test_campaigns = [3, 4]
_test_rate = 0.8
_cv_valid_rate = 0.5


class EM99(rw.score_types.BaseScoreType):
    """99% error margin (EM99) score. Measures the required
    margin in terms of the ratio of the true and predicted
    values to cover 99% of all cases."""

    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='EM99', precision=2, quant=0.99, eps=1e-8):
        self.name = name
        self.precision = precision
        self.quant = quant
        self.eps = eps

    def __call__(self, y_true, y_pred):
        if (y_pred < 0).any():
            return self.worst
        ratio_err = np.array(
            [(p + self.eps) / t for y_hat, y in zip(y_pred, y_true)
             for p, t in zip(y_hat, y) if t != 0])
        # sorted absolute value of mw2dB ratio err
        score = np.percentile(
            np.abs(10 * np.log10(ratio_err)), 100 * self.quant)
        return score


class MEM(rw.score_types.BaseScoreType):
    """Maximum error margin score. Measures the required
    margin in terms of the ratio of the true and predicted
    values to cover all cases. The same as EM100."""

    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='MEM', precision=2, eps=1e-8):
        self.name = name
        self.precision = precision
        self.eps = eps

    def __call__(self, y_true, y_pred):
        if (y_pred < 0).any():
            return self.worst

        ratio_err = np.array(
            [(p + self.eps) / t for y_hat, y in zip(y_pred, y_true)
             for p, t in zip(y_hat, y) if t != 0])
        # sorted absolute value of mw2dB ratio err
        score = np.max(
            np.abs(10 * np.log10(ratio_err)))
        return score


class ONRMSE(rw.score_types.BaseScoreType):
    """Optical network root-mean-square error. Measures the RMSE
     between the true and predicted values of all on channels."""

    is_lower_the_better = True
    minimum = 0.0
    maximum = float('inf')

    def __init__(self, name='ONRMSE', precision=2):
        self.name = name
        self.precision = precision

    def __call__(self, y_true, y_pred):
        on_y_true = np.array([t for y in y_true for t in y if t != 0])
        on_y_pred = np.array([p for y_hat, y in zip(y_pred, y_true) for p, t in zip(y_hat, y) if t != 0])

        if (on_y_pred < 0).any():
            return self.worst

        return np.sqrt(np.mean(np.square(on_y_true - on_y_pred)))


workflow = rw.workflows.Regressor()
Predictions = rw.prediction_types.make_regression(list(range(_NB_CHANNELS)))
score_types = [
    EM99(precision=3),
    ONRMSE(name='RMSE', precision=4),
    MEM(precision=2),
]


def _read_data(path, campaign):
    data_path = os.path.join(path, 'data')
    with open(os.path.join(data_path, f'c{campaign}', 'X.pkl'), 'rb') as f:
        X = pickle.load(f)
        # add campaign index as last column to X
        X = np.array([np.append(x, campaign) for x in X])
    with open(os.path.join(data_path, f'c{campaign}', 'y.pkl'), 'rb') as f:
        y = pickle.load(f)
    return X, y


# Select full cascades from the data: only instances with maximum
# number of modules
def _full_cascade_mask(X):
    lengths = list(map(len, X[:, 0]))  # first column is list of modules
    max_length = np.max(lengths)
    return lengths == max_length


def _train_test_indices(X):
    # random_state must be fixed to avoid leakage
    return train_test_split(
        range(len(X)), test_size=_test_rate, random_state=51)


# Load only full cascades from the test campaigns. is_test_int = 1 of loading 
# test, 0 if loading the portion of the test that goes in train.
def _load_test(path, is_test_int):
    Xs = []
    ys = []
    for campaign in _test_campaigns:
        X, y = _read_data(path, campaign)
        mask = _full_cascade_mask(X)  # only full cascades in test set
        test_is = _train_test_indices(X[mask])[is_test_int]
        Xs.append(X[mask][test_is])
        ys.append(y[mask][test_is])
    return np.concatenate(Xs), np.concatenate(ys)


def get_train_data(path='.'):
    Xs = []
    ys = []
    for campaign in _train_campaigns:
        X, y = _read_data(path, campaign)
        Xs.append(X)
        ys.append(y)
    # adding (1 - _test_rate) of the test full cascades to the training data
    X_test_in_train, y_test_in_train = _load_test(path, is_test_int=0)
    Xs.append(X_test_in_train)
    ys.append(y_test_in_train)
    return np.concatenate(Xs), np.concatenate(ys)


def get_test_data(path='.'):
    return _load_test(path, is_test_int=1)


def get_cv(X, y):
    train_campaigns_is = np.array(
        [i for i, x in enumerate(X) if x[-1] in _train_campaigns])
    test_campaigns_is = np.array(
        [i for i, x in enumerate(X) if x[-1] in _test_campaigns])
    cv_train = ShuffleSplit(
        n_splits=8, test_size=_cv_valid_rate, random_state=42).split(
        train_campaigns_is)
    cv_test = ShuffleSplit(
        n_splits=8, test_size=_cv_valid_rate, random_state=61).split(
        test_campaigns_is)
    for (t_is, v_is), (tt_is, tv_is) in zip(list(cv_train), list(cv_test)):
        # training is both subcascades and part of the test full cascades
        # validation is only test full cascades
        train_is = np.concatenate(
            (train_campaigns_is[t_is], test_campaigns_is[tt_is]))
        valid_is = test_campaigns_is[tv_is]
        yield train_is, valid_is
        # TODO remove when tests done
        # yield train_campaigns_is[t_is], test_campaigns_is[tt_is])
