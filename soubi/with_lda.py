import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "avito")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
PRED_TRAIN = os.path.join(OUTPUT_DIR, "pred_train_next.csv")
GAZOU_TRAIN = os.path.join(OUTPUT_DIR, "image_train.csv")
from avito.common import filename_getter
LDA_DENSE_S_TRAIN, LDA_DESC_S_TRAIN, LDA_TITLE_S_TRAIN = filename_getter.get_lda_filename(OUTPUT_DIR, "stem", "train")
LDA_DENSE_S_TEST, LDA_DESC_S_TEST, LDA_TITLE_S_TEST = filename_getter.get_lda_filename(OUTPUT_DIR, "stem", "test")

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
lgb_col = column_selector.get_predict_col()

train = dd.read_csv(PRED_TRAIN).compute()
gazou = dd.read_csv(GAZOU_TRAIN).compute()
gazou["image"] = gazou["image"].apply(lambda w: w.replace(".jpg", ""))
train = pd.merge(train, gazou, on="image", how="left")
timer.time("load csv in ")

train_y = train["deal_probability"]
train_x = train[predict_col]
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
