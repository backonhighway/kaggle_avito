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
OUTPUT_PRED_V71 = os.path.join(SUBMISSION, "submission71v.csv")
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
gazou = dd.read_csv(GAZOU_TRAIN).compute()
gazou["image"] = gazou["image"].apply(lambda w: w.replace(".jpg", ""))
train = pd.merge(train, gazou, on="image", how="left")
desc_train = scipy.sparse.load_npz(DENSE_TF_TRAIN)
title_train = scipy.sparse.load_npz(TITLE_CNT_TRAIN)

test = dd.read_csv(PRED_TEST).compute()
gazou = dd.read_csv(GAZOU_TEST).compute()
gazou["image"] = gazou["image"].apply(lambda w: w.replace(".jpg", ""))
test = pd.merge(test, gazou, on="image", how="left")
desc_test = scipy.sparse.load_npz(DENSE_TF_TEST)
title_test = scipy.sparse.load_npz(TITLE_CNT_TEST)
timer.time("load csv in ")


train_y = train["deal_probability"]
train_x = train[predict_col]
train_x = scipy.sparse.hstack([scipy.sparse.csr_matrix(train_x), desc_train, title_train])
# max_prob = train.ix[idx_valid]["parent_max_deal_prob"]
test_x = test[predict_col]
test_x = scipy.sparse.hstack([scipy.sparse.csr_matrix(test_x), desc_test, title_test])

timer.time("prepare train in ")
lgb = pocket_lgb.GoldenLgb()
model = lgb.do_train_stack(train_x, train_y, lgb_col)
lgb.show_feature_importance(model)
#exit(0)

# max_map = train.groupby("parent_category_name")["deal_probability"].agg("max").reset_index()
# print(max_map)
# y_pred = model.predict(X_valid)
# train = train.ix[idx_valid]
# train["pred"] = y_pred
# max_pred = train.groupby("parent_category_name")["pred"].agg("max").reset_index()
# print(max_pred)

timer.time("end train in ")
timer.time("done validation in ")

y_pred = model.predict(test_x)
y_pred = np.clip(y_pred, 0.0, 1.0)
submission = pd.DataFrame()
submission["item_id"] = test["item_id"]
submission["deal_probability"] = y_pred
submission.to_csv(OUTPUT_PRED_V71, index=False)

print(train["deal_probability"].describe())
logger.info(submission.describe())
print(submission.describe())
timer.time("done submission in ")

