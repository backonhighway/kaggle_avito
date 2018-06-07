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

import pandas as pd
import numpy as np
import scipy.sparse
import gc
from sklearn import model_selection
from dask import dataframe as dd
from avito.common import csv_loader, column_selector, pocket_lgb, pocket_timer, pocket_logger
from avito.fe import additional_fe

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
dtypes = csv_loader.get_featured_dtypes()
predict_col = column_selector.get_predict_col()
lgb_col = column_selector.get_pred_tf_col()

train = dd.read_csv(PRED_TRAIN).compute()
test = dd.read_csv(PRED_TEST).compute()
timer.time("load csv in ")

lgb = pocket_lgb.GoldenLgb()
# model = lgb.do_train_avito(X_train, X_valid, y_train, y_valid, lgb_col)

submission = pd.DataFrame()
submission["item_id"] = test["item_id"]
submission["deal_probability"] = test["item_id"]
submission.to_csv(OUTPUT_PRED, index=False)
logger.info(submission.describe())
print(submission.describe())
timer.time("done submission in ")
