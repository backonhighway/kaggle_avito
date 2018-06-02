import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "avito")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
SUBMISSION = os.path.join(APP_ROOT, "submission")
PRED_TRAIN = os.path.join(OUTPUT_DIR, "pred_train.csv")
PRED_TEST = os.path.join(OUTPUT_DIR, "pred_test.csv")
OUTPUT_PRED = os.path.join(SUBMISSION, "submission.csv")
MODEL_FILE = os.path.join(SUBMISSION, "pred_model.txt")
from avito.common import filename_getter
DESC_TF_COLS, DESC_TF_TRAIN, DESC_TF_TEST = filename_getter.get_filename(OUTPUT_DIR, "desc", "tf")
TITLE_TF_COLS, TITLE_TF_TRAIN, TITLE_TF_TEST = filename_getter.get_filename(OUTPUT_DIR, "title", "tf")

import pandas as pd
import numpy as np
import scipy.sparse
import gc
from sklearn import model_selection
from dask import dataframe as dd
from avito.common import csv_loader, column_selector, pocket_lgb, pocket_timer, pocket_logger

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
dtypes = csv_loader.get_featured_dtypes()
predict_col = column_selector.get_predict_col()
lgb_col = column_selector.get_pred_tf_col()

train = dd.read_csv(PRED_TRAIN).compute()
desc_train = scipy.sparse.load_npz(DESC_TF_TRAIN)
title_train = scipy.sparse.load_npz(TITLE_TF_TRAIN)

test = dd.read_csv(PRED_TEST).compute()
desc_test = scipy.sparse.load_npz(DESC_TF_TEST)
title_test = scipy.sparse.load_npz(TITLE_TF_TEST)
timer.time("load csv in ")

train_y = train["deal_probability"]
train_x = train[predict_col]
train_x = scipy.sparse.hstack([scipy.sparse.csr_matrix(train_x), desc_train, title_train])
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(train_x, train_y, test_size=0.2, random_state=99)

timer.time("prepare train in ")
lgb = pocket_lgb.GoldenLgb()
model = lgb.do_train_avito(X_train, X_valid, y_train, y_valid, lgb_col)
lgb.show_feature_importance(model)
#exit(0)

del train, X_train, X_valid, y_train, y_valid
gc.collect()
timer.time("end train in ")

test_x = test[predict_col]
test_x = scipy.sparse.hstack([scipy.sparse.csr_matrix(test_x), desc_test, title_test])
y_pred = model.predict(test_x)
y_pred = np.clip(y_pred, 0.0, 1.0)
submission = pd.DataFrame()
submission["item_id"] = test["item_id"]
submission["deal_probability"] = y_pred
submission.to_csv(OUTPUT_PRED, index=False)
logger.info(submission.describe())
print(submission.describe())
timer.time("done submission in ")
