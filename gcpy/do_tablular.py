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
PERIOD_TRAIN = os.path.join(INPUT_DIR, "periods_train.csv")
PERIOD_TEST = os.path.join(INPUT_DIR, "periods_test.csv")
PRED_TRAIN = os.path.join(OUTPUT_DIR, "pred_train.csv")
PRED_TEST = os.path.join(OUTPUT_DIR, "pred_test.csv")

import numpy as np
import pandas as pd
import gc
from sklearn import preprocessing
from avito.common import pocket_timer, pocket_logger, column_selector
from avito.fe import active_fe

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
#dtypes = csv_loader.get_featured_dtypes()
#predict_col = column_selector.get_predict_col()

train = pd.read_csv(ORG_TRAIN, nrows=1000*10)
test = pd.read_csv(ORG_TEST, nrows=1000*10)
train_active = pd.read_csv(ACTIVE_TRAIN, nrows=1000*10)
test_active = pd.read_csv(ACTIVE_TEST, nrows=1000*10)
train_period = pd.read_csv(PERIOD_TRAIN, nrows=1000*10)
test_period = pd.read_csv(PERIOD_TEST, nrows=1000*10)
#train = pd.read_csv(ORG_TRAIN)
#test = pd.read_csv(ORG_TEST)
#train_active = pd.read_csv(ACTIVE_TRAIN)
#test_active = pd.read_csv(ACTIVE_TEST)
#train_period = pd.read_csv(PERIOD_TRAIN)
#test_period = pd.read_csv(PERIOD_TEST)

print(train.describe())
timer.time("read csv")
print("-"*40)

train, test = active_fe.doit(train, test, train_active, test_active, train_period, test_period, timer)

whole_col = column_selector.get_whole_col()
test_col = column_selector.get_test_col()
train = train[whole_col]
test = test[test_col]
# print(train.head())
# print(train.describe())
# print(test.head())
# print(test.describe())

train.to_csv(PRED_TRAIN)
test.to_csv(PRED_TEST)
