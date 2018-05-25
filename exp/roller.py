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

# train = pd.read_csv(ORG_TRAIN)
train = pd.read_csv(ORG_TRAIN)
# test = pd.read_csv(ORG_TEST)
print(train.describe())
# train = train[train["parent_category_name"] == "Услуги"]
#print(train.head())
train["image_cat"] = train["image_top_1"].fillna(-1)
timer.time("start timing")
grouped = train.groupby(["category_name", "image_cat"])["price"].rolling(4, min_periods=1).mean()
print(grouped)
timer.time("normal")
train["roller"] = train.groupby(["category_name", "image_cat"])["price"].\
   transform(lambda g: g.rolling(window=4).mean())
timer.time("transformed")
exit(0)
print(train.describe())
