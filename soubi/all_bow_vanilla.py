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
DENSE_TF_COLS, DENSE_TF_TRAIN, DENSE_TF_TEST = filename_getter.get_filename(OUTPUT_DIR, "stem_title_desc", "tf")
TITLE_CNT_COLS, TITLE_CNT_TRAIN, TITLE_CNT_TEST = filename_getter.get_filename(OUTPUT_DIR, "stem_title", "cnt")
DENSE_CNT15_COLS, DENSE_CNT15_TRAIN, DENSE_CNT15_TEST = \
    filename_getter.get_filename(OUTPUT_DIR, "stem_cnt15_dense", "cnt")
DESC_CNT15_COLS, DESC_CNT15_TRAIN, DESC_CNT15_TEST = \
    filename_getter.get_filename(OUTPUT_DIR, "stem_cnt15_desc", "cnt")
TITLE_CNT15_COLS, TITLE_CNT15_TRAIN, TITLE_CNT15_TEST = \
    filename_getter.get_filename(OUTPUT_DIR, "stem_cnt15_title", "cnt")
VDESC_TF_COLS, VDESC_TF_TRAIN, VDESC_TF_TEST = filename_getter.get_filename(OUTPUT_DIR, "desc", "tf")
VTITLE_TF_COLS, VTITLE_TF_TRAIN, VTITLE_TF_TEST = filename_getter.get_filename(OUTPUT_DIR, "title", "tf")
VDENSE_TF_COLS, VDENSE_TF_TRAIN, VDENSE_TF_TEST = filename_getter.get_filename(OUTPUT_DIR, "title_desc", "tf")

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
files = [DENSE_TF_COLS, DESC_TF_COLS, TITLE_TF_COLS,
         DENSE_CNT15_COLS, DESC_CNT15_COLS, TITLE_CNT15_COLS,
         VDENSE_TF_COLS, VDESC_TF_COLS, VTITLE_TF_COLS,]
names = ["dense_tf_", "desc_tf_", "title_tf",
         "dense_cnt15_", "desc_cnt15_", "title_cnt15_",
         "vdense_tf_", "vdesc_tf_", "vtitle_tf",]
lgb_col = column_selector.get_cols_from_files(files, names)

train = dd.read_csv(PRED_TRAIN).compute()
gazou = dd.read_csv(GAZOU_TRAIN).compute()
#gazou["image"] = gazou["image"].apply(lambda w: w.replace(".jpg", ""))
train = pd.merge(train, gazou, on="image", how="left")
dense_tf_train = scipy.sparse.load_npz(DENSE_TF_TRAIN)
desc_tf_train = scipy.sparse.load_npz(DESC_TF_TRAIN)
title_tf_train = scipy.sparse.load_npz(TITLE_TF_TRAIN)
dense_cnt_train = scipy.sparse.load_npz(DENSE_CNT15_TRAIN)
desc_cnt_train = scipy.sparse.load_npz(DESC_CNT15_TRAIN)
title_cnt_train = scipy.sparse.load_npz(TITLE_CNT15_TRAIN)
title_cnt_all_train = scipy.sparse.load_npz(TITLE_CNT_TRAIN)
vdense_tf_train = scipy.sparse.load_npz(VDENSE_TF_TRAIN)
vdesc_tf_train = scipy.sparse.load_npz(VDESC_TF_TRAIN)
vtitle_tf_train = scipy.sparse.load_npz(VTITLE_TF_TRAIN)
timer.time("load csv in ")

train_y = train["deal_probability"]
train_x = train[predict_col]
the_stack = [scipy.sparse.csr_matrix(train_x), dense_tf_train, desc_tf_train, title_tf_train,
             dense_cnt_train, desc_cnt_train, title_cnt_train, vdense_tf_train, vdesc_tf_train, vtitle_tf_train,]
train_x = scipy.sparse.hstack(the_stack)
print(train_x.shape)
train_idx = train.index.values
X_train, X_valid, y_train, y_valid, idx_train, idx_valid = \
    model_selection.train_test_split(train_x, train_y, train_idx, test_size=0.2, random_state=99)
max_prob = train.ix[idx_valid]["parent_max_deal_prob"]

timer.time("prepare train in ")
lgb = pocket_lgb.GoldenLgb()
model = lgb.do_train_avito(X_train, X_valid, y_train, y_valid, lgb_col)
lgb.show_feature_importance(model)
#exit(0)

max_map = train.groupby("parent_category_name")["deal_probability"].agg("max").reset_index()
print(max_map)
y_pred = model.predict(X_valid)
train = train.ix[idx_valid]
train["pred"] = y_pred
max_pred = train.groupby("parent_category_name")["pred"].agg("max").reset_index()
print(max_pred)

# del train, X_train, X_valid, y_train, y_valid
# gc.collect()
timer.time("end train in ")
validator = holdout_validator.HoldoutValidator(model, X_valid, y_valid, max_prob)
validator.validate()
# validator.output_prediction(PREDICTION)

timer.time("done validation in ")
