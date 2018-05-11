import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "avito")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
ORG_TRAIN = os.path.join(INPUT_DIR, "train.csv")
PRED_TRAIN = os.path.join(OUTPUT_DIR, "pred_train.csv")

import numpy as np
import pandas as pd
import gc
from sklearn import preprocessing
from avito.common import pocket_timer, pocket_logger, column_selector

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
#dtypes = csv_loader.get_featured_dtypes()
#predict_col = column_selector.get_predict_col()
whole_col = column_selector.get_whole_col()

train = pd.read_csv(ORG_TRAIN)
print(train.describe())
timer.time("read csv")
print("-"*40)
# "item_id", "user_id", title, description
cat_cols = ["region", "city", "parent_category_name", "category_name",
            "param_1", "param_2", "param_3", "user_type"]
for col in cat_cols:
    print(col)
    le = preprocessing.LabelEncoder()
    train[col] = le.fit_transform(train[col].astype("str"))
    # lbl.fit(list(train_df[col].values.astype('str')) + list(test_df[col].values.astype('str')))

train["activation_date"] = pd.to_datetime(train["activation_date"])
train["activation_dow"] = train["activation_date"].dt.dayofweek
train["activation_day"] = train["activation_date"].dt.day

train = train[whole_col]
print(train.head())
print(train.info())
print(train.describe())

train.to_csv(PRED_TRAIN)
