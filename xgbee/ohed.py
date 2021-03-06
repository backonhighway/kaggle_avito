import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "avito")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
PRED_TRAIN = os.path.join(OUTPUT_DIR, "pred_train.csv")
GAZOU_TRAIN = os.path.join(OUTPUT_DIR, "image_train.csv")
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
from avito.common import csv_loader, column_selector, pocket_xgb, pocket_timer, pocket_logger, holdout_validator
from avito.fe import additional_fe

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
dtypes = csv_loader.get_featured_dtypes()
predict_col = column_selector.get_predict_col()
lgb_col = column_selector.get_stem_col()
lgb_col = [c.replace(" ", "_") for c in lgb_col]
# tail = lgb_col[-5:]
# print(tail)

train = dd.read_csv(PRED_TRAIN).compute()
gazou = dd.read_csv(GAZOU_TRAIN).compute()
#gazou["image"] = gazou["image"].apply(lambda w: w.replace(".jpg", ""))
train = pd.merge(train, gazou, on="image", how="left")
desc_train = scipy.sparse.load_npz(DENSE_TF_TRAIN)
title_train = scipy.sparse.load_npz(TITLE_CNT_TRAIN)
timer.time("load csv in ")

cat_col = [
    "region", "city", "parent_category_name", "category_name",
    "param_1", "param_2", "param_3", "param_all", "image_top_1", "user_type"
]
train = pd.get_dummies(data=train, prefix=cat_col, dummy_na=True, columns=cat_col, sparse=True,)
predict_col= [c for c in predict_col if c not in cat_col]
#train.drop(cat_col, axis=1, inplace=True)
train.fillna(-999, inplace=True)

train_y = train["deal_probability"]
train_x = train[predict_col]
train_x = scipy.sparse.hstack([scipy.sparse.csr_matrix(train_x), desc_train])
train_idx = train.index.values
X_train, X_valid, y_train, y_valid, idx_train, idx_valid = \
    model_selection.train_test_split(train_x, train_y, train_idx, test_size=0.2, random_state=99)
max_prob = train.ix[idx_valid]["parent_max_deal_prob"]

timer.time("prepare train in ")
xgb = pocket_xgb.GoldenXgb()
model = xgb.do_train_avito(X_train, X_valid, y_train, y_valid, lgb_col)
#xgb.show_feature_importance(model)
#exit(0)

max_map = train.groupby("parent_category_name")["deal_probability"].agg("max").reset_index()
print(max_map)
y_pred = xgb.do_predict(model, X_valid)
train = train.ix[idx_valid]
train["pred"] = y_pred
max_pred = train.groupby("parent_category_name")["pred"].agg("max").reset_index()
print(max_pred)

# del train, X_train, X_valid, y_train, y_valid
# gc.collect()
timer.time("end train in ")

