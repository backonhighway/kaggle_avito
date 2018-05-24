import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "avito")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
ORG_TRAIN = os.path.join(INPUT_DIR, "train.csv")
ORG_TEST = os.path.join(INPUT_DIR, "test.csv")
OUTPUT_PRED = os.path.join(OUTPUT_DIR, "submission.csv")
PRED_TRAIN = os.path.join(OUTPUT_DIR, "pred_train.csv")
PRED_TEST = os.path.join(OUTPUT_DIR, "pred_test.csv")

import numpy as np
import pandas as pd
import gc
from sklearn import preprocessing
from avito.common import pocket_timer, pocket_logger, column_selector

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
#dtypes = csv_loader.get_featured_dtypes()
#predict_col = column_selector.get_predict_col()

train = pd.read_csv(ORG_TRAIN, nrows=1000*100)
# test = pd.read_csv(ORG_TEST)

print(train.info())

train["user_item_count"] = train.groupby("user_id")["item_id"].transform("count")
train["user_max_seq"] = train.groupby("user_id")["item_seq_number"].transform("max")
train["is_same"] = np.where(train["user_item_count"] == train["user_max_seq"], 1, 0)

print(train["user_item_count"].describe())
print(train["item_seq_number"].describe())
print(train["user_max_seq"].describe())
print(train["is_same"].describe())

wowy = train.groupby("user_max_seq")["deal_probability"].agg("mean").reset_index()
print(wowy.describe())
import matplotlib.pyplot as plt
#plt.hist(wowy, bins=np.arange(0, 410, 10))
plt.scatter(wowy["user_max_seq"], wowy["deal_probability"])
plt.show()
