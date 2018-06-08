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

import numpy as np
import pandas as pd
import gc
from dask import dataframe as dd
from sklearn import preprocessing
from avito.common import pocket_timer, pocket_logger, column_selector

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
#dtypes = csv_loader.get_featured_dtypes()
#predict_col = column_selector.get_predict_col()

train = pd.read_csv(ORG_TRAIN)
test = pd.read_csv(ORG_TEST)
trainA = pd.read_csv(ACTIVE_TRAIN)
# testA = dd.read_csv(ACTIVE_TEST).compute()

# trainA.to_csv(ACTIVE_TRAIN1, index=False)
# testA.to_csv(ACTIVE_TEST1, index=False)

print(train["item_seq_num"].describe())
print(test["item_seq_num"].describe())

all_df = pd.concat([train, trainA], axis=1)

count = all_df.groupby("user_id")["item_id"].count().reset_index()
print(count.describe())
count = all_df.groupby(["user_id", "parent_category_name", "category_name"])["item_id"].count().reset_index()
print(count.describe())

all_df["seq_diff"] = all_df.groupby("user_id")["item_seq_num"].shift(1)
all_df["seq_diff"] = all_df["seq_diff"] - all_df["item_seq_num"]
seq_diff = all_df.groupby("seq_diff")["deal_probability"].mean().reset_index()
print(seq_diff)

print("done ok")









