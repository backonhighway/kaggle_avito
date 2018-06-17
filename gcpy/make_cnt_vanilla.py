import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "avito")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
ORG_TRAIN = os.path.join(INPUT_DIR, "train.csv")
ORG_TEST = os.path.join(INPUT_DIR, "test.csv")
from avito.common import filename_getter
DENSE_CNT15_COLS, DENSE_CNT15_TRAIN, DENSE_CNT15_TEST = \
    filename_getter.get_filename(OUTPUT_DIR, "vanilla_cnt15_dense", "cnt")
DESC_CNT15_COLS, DESC_CNT15_TRAIN, DESC_CNT15_TEST = \
    filename_getter.get_filename(OUTPUT_DIR, "vanilla_cnt15_desc", "cnt")
TITLE_CNT15_COLS, TITLE_CNT15_TRAIN, TITLE_CNT15_TEST = \
    filename_getter.get_filename(OUTPUT_DIR, "vanilla_cnt15_title", "cnt")


import numpy as np
import pandas as pd
import scipy.sparse
import gc
from avito.common import pocket_timer, pocket_logger, column_selector
from avito.fe import lda_cnt_bow

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
#dtypes = csv_loader.get_featured_dtypes()
#predict_col = column_selector.get_predict_col()

# train = pd.read_csv(ORG_TRAIN, nrows=1000*10)
# test = pd.read_csv(ORG_TEST, nrows=1000*10)
train = pd.read_csv(ORG_TRAIN)
test = pd.read_csv(ORG_TEST)
timer.time("read csv")
print("-"*40)
train_col = ["item_id", "title", "description"]
test_col = ["item_id", "title", "description"]
train = train[train_col]
test = test[test_col]
train["title_desc"] = train["title"] + " " + train["description"]
test["title_desc"] = test["title"] + " " + test["description"]

temp_train, temp_test, temp_names= lda_cnt_bow.make_dense_cnt(train, test)
timer.time("done dense cnt")
lda_cnt_bow.save_sparsed((DENSE_CNT15_COLS, DENSE_CNT15_TRAIN, DENSE_CNT15_TEST), (temp_names, temp_train, temp_test))
timer.time("saved dense cnt")

temp_train, temp_test, temp_names= lda_cnt_bow.make_desc_cnt(train, test)
timer.time("done desc cnt")
lda_cnt_bow.save_sparsed((DESC_CNT15_COLS, DESC_CNT15_TRAIN, DESC_CNT15_TEST), (temp_names, temp_train, temp_test))
timer.time("saved desc cnt")

temp_train, temp_test, temp_names= lda_cnt_bow.make_title_cnt(train, test)
timer.time("done title cnt")
lda_cnt_bow.save_sparsed((TITLE_CNT15_COLS, TITLE_CNT15_TRAIN, TITLE_CNT15_TEST), (temp_names, temp_train, temp_test))
timer.time("saved title cnt")