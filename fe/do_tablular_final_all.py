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
PRED_TRAIN = os.path.join(OUTPUT_DIR, "pred_train_all.csv")
PRED_TEST = os.path.join(OUTPUT_DIR, "pred_test_all.csv")

import numpy as np
import pandas as pd
import gc
from sklearn import preprocessing
from avito.common import pocket_timer, pocket_logger, column_selector
from avito.fe import final_fe_all

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
#dtypes = csv_loader.get_featured_dtypes()
# predict_col = column_selector.get_predict_col()

# train = pd.read_csv(ORG_TRAIN, nrows=1000*10)
# test = pd.read_csv(ORG_TEST, nrows=1000*10)
# train_active = pd.read_csv(ACTIVE_TRAIN, nrows=1000*10)
# test_active = pd.read_csv(ACTIVE_TEST, nrows=1000*10)
# train_period = pd.read_csv(PERIOD_TRAIN, nrows=1000*10, parse_dates=['date_from', 'date_to'])
# test_period = pd.read_csv(PERIOD_TEST, nrows=1000*10, parse_dates=['date_from', 'date_to'])
train = pd.read_csv(ORG_TRAIN)
test = pd.read_csv(ORG_TEST)
train_active = pd.read_csv(ACTIVE_TRAIN)
test_active = pd.read_csv(ACTIVE_TEST)
train_period = pd.read_csv(PERIOD_TRAIN, parse_dates=['date_from', 'date_to'])
test_period = pd.read_csv(PERIOD_TEST, parse_dates=['date_from', 'date_to'])
print(train_active.shape)

timer.time("read csv")
print("-"*40)

train, test = final_fe_all.doit(train, test, train_active, test_active, train_period, test_period, timer)
print(train.describe())
print(list(train))

# whole_col = column_selector.get_whole_col()
# test_col = column_selector.get_test_col()
# train = train[whole_col]
# test = test[test_col]

# print(train.head())
# print(train.describe())
# print(test.head())
# print(test.describe())


drop_col = ["title", "description"]
train.drop(drop_col, axis=1, inplace=True)
test.drop(drop_col, axis=1, inplace=True)

train.to_csv(PRED_TRAIN, index=False)
test.to_csv(PRED_TEST, index=False)

