import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "avito")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
SUBMISSION = os.path.join(APP_ROOT, "submission")
PRED_TRAIN = os.path.join(OUTPUT_DIR, "pred_train.csv")
PRED_TEST = os.path.join(OUTPUT_DIR, "pred_test.csv")
GAZOU_TRAIN = os.path.join(OUTPUT_DIR, "image_train.csv")
GAZOU_TEST = os.path.join(OUTPUT_DIR, "image_test.csv")
OUTPUT_PRED = os.path.join(SUBMISSION, "submission.csv")
MODEL_FILE = os.path.join(SUBMISSION, "pred_model.txt")

import pandas as pd
import numpy as np
import scipy.sparse
import gc
from sklearn import preprocessing
from dask import dataframe as dd
from avito.common import csv_loader, column_selector, pocket_lgb, pocket_timer, pocket_logger, holdout_validator
from avito.fe import additional_fe

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
# train = pd.read_csv(PRED_TRAIN, nrows=1000*100)
# test = pd.read_csv(PRED_TEST, nrows=1000*100)
train = dd.read_csv(PRED_TRAIN).compute()
test = dd.read_csv(PRED_TEST).compute()
gazou_train = dd.read_csv(GAZOU_TRAIN).compute()
gazou_test = dd.read_csv(GAZOU_TEST).compute()
timer.time("load csv")

scaler = preprocessing.MinMaxScaler()
gazou_train["image_timestamp_scaled"] = scaler.fit_transform(gazou_train[["image_timestamp"]])
gazou_test["image_timestamp_scaled"] = scaler.fit_transform(gazou_test[["image_timestamp"]])

# print(gazou_train["image_timestamp"].describe())
# min_val = gazou_train["image_timestamp"].min()
# print(min_val)
# gazou_train["image_timestamp"] = gazou_train["image_timestamp"] - min_val
# print(gazou_train["image_timestamp"].describe())
#
# print(gazou_test["image_timestamp"].describe())
# min_val = gazou_test["image_timestamp"].min()
# print(min_val)
# gazou_test["image_timestamp"] = gazou_test["image_timestamp"] - min_val
# print(gazou_test["image_timestamp"].describe())

# train = additional_fe.get_user_history(train)
# print(train["user_deal_prob"].describe())
# test = additional_fe.get_test_user_history(train, test)
# print(test["user_deal_prob"].describe())
timer.time("done fe")

gazou_train.to_csv(GAZOU_TRAIN, index=False)
gazou_test.to_csv(GAZOU_TEST, index=False)
timer.time("output csv")