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
from sklearn import preprocessing
from avito.common import pocket_timer, pocket_logger, column_selector

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
#dtypes = csv_loader.get_featured_dtypes()
#predict_col = column_selector.get_predict_col()

train = pd.read_csv(ORG_TRAIN)
test = pd.read_csv(ORG_TEST)
trainA = pd.read_csv(ACTIVE_TRAIN, nrows=1000*1000)
testA = pd.read_csv(ACTIVE_TEST, nrows=1000*1000)

# trainA.to_csv(ACTIVE_TRAIN1, index=False)
# testA.to_csv(ACTIVE_TEST1, index=False)

print(len(set(train["item_id"])))
print(len(set(trainA["item_id"])))
#train["in_active"] = np.where(train["item_id"].isin(trainA["item_id"]), 1, 0)
train["in_active"] = np.where(train["item_id"].isin(test["item_id"]), 1, 0)
print(train["in_active"].describe())
train["in_active"] = np.where(train["item_id"].isin(testA["item_id"]), 1, 0)
print(train["in_active"].describe())

print("done ok")









