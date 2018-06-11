import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "avito")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
ORG_TRAIN = os.path.join(INPUT_DIR, "train.csv")
ORG_TEST = os.path.join(INPUT_DIR, "test.csv")
STEM_TRAIN = os.path.join(OUTPUT_DIR, "stem_train.csv")
STEM_TEST = os.path.join(OUTPUT_DIR, "stem_test.csv")
PRED_TRAIN = os.path.join(OUTPUT_DIR, "pred_train.csv")
PRED_TEST = os.path.join(OUTPUT_DIR, "pred_test.csv")
from avito.common import filename_getter
DESC_TF_COLS, DESC_TF_TRAIN, DESC_TF_TEST = filename_getter.get_filename(OUTPUT_DIR, "stem_desc", "tf")
TITLE_TF_COLS, TITLE_TF_TRAIN, TITLE_TF_TEST = filename_getter.get_filename(OUTPUT_DIR, "stem_title", "tf")
TITLE_CNT_COLS, TITLE_CNT_TRAIN, TITLE_CNT_TEST = filename_getter.get_filename(OUTPUT_DIR, "stem_title", "cnt")
DENSE_TF_COLS, DENSE_TF_TRAIN, DENSE_TF_TEST = filename_getter.get_filename(OUTPUT_DIR, "stem_title_desc", "tf")
DENSE_CNT_COLS, DENSE_CNT_TRAIN, DENSE_CNT_TEST = filename_getter.get_filename(OUTPUT_DIR, "stem_title_desc", "cnt")


import numpy as np
import pandas as pd
import scipy.sparse
import gc
from avito.common import pocket_timer, pocket_logger, column_selector
from avito.fe import big_bow

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
#dtypes = csv_loader.get_featured_dtypes()
#predict_col = column_selector.get_predict_col()

# train = pd.read_csv(ORG_TRAIN, nrows=1000*10)
# test = pd.read_csv(ORG_TEST, nrows=1000*10)
train = pd.read_csv(STEM_TRAIN)
test = pd.read_csv(STEM_TEST)
timer.time("read csv")
print("-"*40)
train.columns=["item_id", "title", "description"]
test.columns=["item_id", "title", "description"]
train["title_desc"] = train["title"] + " " + train["description"]
test["title_desc"] = test["title"] + " " + test["description"]
# print(train.head()["title"])
# print(train.head()["description"])
# print(train.head()["title_desc"])

desc_train, desc_test, desc_tf_names = big_bow.make_dense_cnt(train, test)
timer.time("done dense cnt")
big_bow.save_sparsed((DESC_TF_COLS, DESC_TF_TRAIN, DESC_TF_TEST), (desc_tf_names, desc_train, desc_test))
timer.time("saved dense cnt")
