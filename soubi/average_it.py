import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "avito")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
SUBMISSION = os.path.join(APP_ROOT, "submission")
PRED_TRAIN = os.path.join(OUTPUT_DIR, "pred_train.csv")
PRED_TEST = os.path.join(OUTPUT_DIR, "pred_test.csv")
GAZOU_TRAIN = os.path.join(OUTPUT_DIR, "image_train.csv")
GAZOU_TEST = os.path.join(OUTPUT_DIR, "image_test.csv")
OUTPUT_PRED = os.path.join(SUBMISSION, "submission.csv")
OUTPUT_PRED_52 = os.path.join(SUBMISSION, "submission52.csv")
OUTPUT_PRED_54 = os.path.join(SUBMISSION, "submission54.csv")
OUTPUT_PRED_99 = os.path.join(SUBMISSION, "submission99.csv")
MODEL_FILE = os.path.join(SUBMISSION, "pred_model.txt")
from avito.common import filename_getter
DESC_TF_COLS, DESC_TF_TRAIN, DESC_TF_TEST = filename_getter.get_filename(OUTPUT_DIR, "stem_desc", "tf")
TITLE_TF_COLS, TITLE_TF_TRAIN, TITLE_TF_TEST = filename_getter.get_filename(OUTPUT_DIR, "stem_title", "tf")
TITLE_CNT_COLS, TITLE_CNT_TRAIN, TITLE_CNT_TEST = filename_getter.get_filename(OUTPUT_DIR, "stem_title", "cnt")
DENSE_TF_COLS, DENSE_TF_TRAIN, DENSE_TF_TEST = filename_getter.get_filename(OUTPUT_DIR, "stem_title_desc", "tf")

import pandas as pd
import numpy as np
import scipy.sparse
import gc
from sklearn import model_selection
from dask import dataframe as dd
from avito.common import csv_loader, column_selector, pocket_lgb, pocket_timer, pocket_logger, holdout_validator
from avito.fe import additional_fe

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
dtypes = csv_loader.get_featured_dtypes()
predict_col = column_selector.get_predict_col()
lgb_col = column_selector.get_stem_col()

train = dd.read_csv(PRED_TRAIN).compute()
test = dd.read_csv(PRED_TEST).compute()
pred1 = dd.read_csv(OUTPUT_PRED_52).compute()
pred2 = dd.read_csv(OUTPUT_PRED_54).compute()
pred3 = dd.read_csv(OUTPUT_PRED_99).compute()
timer.time("read csv")

submission = pd.DataFrame()
submission["item_id"] = test["item_id"]
submission["deal_probability"] = \
    (pred1["deal_probability"] + pred2["deal_probability"] + pred3["deal_probability"]) / 3
submission.to_csv(OUTPUT_PRED, index=False)

print(train["deal_probability"].describe())
logger.info(submission.describe())
print(submission.describe())
timer.time("done submission in ")