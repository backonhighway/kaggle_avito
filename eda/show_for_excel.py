import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "avito")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
ORG_TRAIN = os.path.join(INPUT_DIR, "train.csv")
ORG_TEST = os.path.join(INPUT_DIR, "test.csv")
ACTIVE_TRAIN = os.path.join(INPUT_DIR, "train_active.csv")
ACTIVE_TEST = os.path.join(INPUT_DIR, "test_active.csv")
OUTPUT_PRED = os.path.join(OUTPUT_DIR, "submission.csv")
PRED_TRAIN = os.path.join(OUTPUT_DIR, "pred_train.csv")
PRED_TEST = os.path.join(OUTPUT_DIR, "pred_test.csv")
ACTIVE_TRAIN1 = os.path.join(INPUT_DIR, "train_active_first1M.csv")
ACTIVE_TEST1 = os.path.join(INPUT_DIR, "test_active_first1M.csv")

SMALL_TRAIN = os.path.join(INPUT_DIR, "train_small.csv")

import numpy as np
import pandas as pd
import gc
from sklearn import preprocessing
from avito.common import pocket_timer, pocket_logger, column_selector

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
#dtypes = csv_loader.get_featured_dtypes()
#predict_col = column_selector.get_predict_col()

train = pd.read_csv(ORG_TRAIN, nrows=1000*10)

param_list = ["param_1", "param_2", "param_3"]
for some_param in param_list:
    train[some_param].fillna("_", inplace=True)
train['param_all'] = train["param_1"] + train["param_2"] + train["param_3"]

cat_cols = ["region", "city", "parent_category_name", "category_name",
            "param_1", "param_2", "param_3", "param_all", "user_type"]
for col in cat_cols:
    print(col)
    le = preprocessing.LabelEncoder()
    #train[col] = le.fit_transform(train[col].astype("str"))
    le.fit(list(train[col].values.astype('str')))
    train[col] = le.transform(train[col].values.astype('str'))
train = train.drop(["title", "description"], axis=1)
print(train.head())

train.to_csv(SMALL_TRAIN, index=False)







