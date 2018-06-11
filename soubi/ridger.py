import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "avito")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
PRED_TRAIN = os.path.join(OUTPUT_DIR, "pred_train.csv")
PRED_TEST = os.path.join(OUTPUT_DIR, "pred_test.csv")
from avito.common import filename_getter
DESC_TF_COLS, DESC_TF_TRAIN, DESC_TF_TEST = filename_getter.get_filename(OUTPUT_DIR, "stem_desc", "tf")
TITLE_TF_COLS, TITLE_TF_TRAIN, TITLE_TF_TEST = filename_getter.get_filename(OUTPUT_DIR, "stem_title", "tf")
TITLE_CNT_COLS, TITLE_CNT_TRAIN, TITLE_CNT_TEST = filename_getter.get_filename(OUTPUT_DIR, "stem_title", "cnt")
DENSE_TF_COLS, DENSE_TF_TRAIN, DENSE_TF_TEST = filename_getter.get_filename(OUTPUT_DIR, "stem_title_desc", "tf")

import pandas as pd
import numpy as np
import scipy.sparse
from dask import dataframe as dd
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn import metrics


class SklearnWrapper(object):
    def __init__(self, clf, seed=0, params=None, seed_bool=True):
        if seed_bool:
            params['random_state'] = seed
        self.clf = clf(**params)

    def train(self, x_train, y_train):
        self.clf.fit(x_train, y_train)

    def predict(self, x):
        return self.clf.predict(x)


def get_oof(clf, x_train, y, x_test, kfold):
    oof_train = np.zeros((train_rows,))
    oof_test = np.zeros((test_rows,))
    oof_test_skf = np.empty((NFOLDS, test_rows))

    i = 0
    for train_index, test_index in kfold:
        print('\nFold {}'.format(i))
        x_tr = x_train[train_index]
        y_tr = y[train_index]
        x_te = x_train[test_index]

        clf.train(x_tr, y_tr)

        oof_train[test_index] = clf.predict(x_te)
        oof_test_skf[i, :] = clf.predict(x_test)
        i = i + 1

    oof_test[:] = oof_test_skf.mean(axis=0)
    return oof_train.reshape(-1, 1), oof_test.reshape(-1, 1)


NFOLDS = 4
SEED = 99
train = dd.read_csv(PRED_TRAIN).compute()
test = dd.read_csv(PRED_TEST).compute()
y = train["deal_probability"]

desc_train = scipy.sparse.load_npz(DENSE_TF_TRAIN)
title_train = scipy.sparse.load_npz(TITLE_CNT_TRAIN)
desc_test = scipy.sparse.load_npz(DENSE_TF_TEST)
title_test = scipy.sparse.load_npz(TITLE_CNT_TEST)
train_bow = scipy.sparse.hstack([desc_train, title_train])
test_bow = scipy.sparse.hstack([desc_test, title_test])

train_rows = train.shape[0]
test_rows = test.shape[0]

kf = KFold(n_splits=NFOLDS, shuffle=True, random_state=SEED).split(train_rows)

ridge_params = {'alpha':30.0, 'fit_intercept':True, 'normalize':False, 'copy_X':True,
                'max_iter':None, 'tol':0.001, 'solver':'auto', 'random_state':SEED}
ridge = SklearnWrapper(clf=Ridge, seed=SEED, params=ridge_params)
ridge_oof_train, ridge_oof_test = get_oof(ridge, train_bow, y, test_bow, kf)

ridge_pred = np.concatenate([ridge_oof_train, ridge_oof_test])
rmse_score = metrics.mean_squared_error(y, ridge_oof_train) ** 0.5
print(rmse_score)