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

import pandas as pd
import numpy as np
import scipy.sparse
import gc
from sklearn import model_selection
from dask import dataframe as dd
from avito.common import csv_loader, column_selector, pocket_lgb, pocket_timer, pocket_logger
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

test = dd.read_csv(PRED_TEST).compute()
gazou = dd.read_csv(GAZOU_TEST).compute()
gazou["image"] = gazou["image"].apply(lambda w: w.replace(".jpg", ""))
test = pd.merge(test, gazou, on="image", how="left")
timer.time("load csv in ")

# train = train[predict_col]
# train = scipy.sparse.hstack([scipy.sparse.csr_matrix(train), desc_train, title_train])
test_x = test[predict_col]

train_y = train["deal_probability"]
train_x = train[predict_col]
timer.time("prepare train in ")

split_num = 4
skf = model_selection.KFold(n_splits=split_num, shuffle=False)
lgb = pocket_lgb.GoldenLgb()
total_score = 0
models = []

for train_index, test_index in skf.split(train):
    X_train, X_test = train_x.ix[train_index], train_x.ix[test_index]
    y_train, y_test = train_y.iloc[train_index], train_y.iloc[test_index]
    model = lgb.do_train_avito(X_train, X_test, y_train, y_test, lgb_col)
    score = model.best_score["valid_0"]["rmse"]
    total_score += score

    models.append(model)

lgb.show_feature_importance(models[0])
print("average score= ", total_score / split_num)
timer.time("end train in ")

# train_rows = train.shape[0]
# test_rows = test.shape[0]
# oof_train = np.zeros((train_rows,))
# oof_pred = np.zeros((test_rows,))
# test_max_value = test["parent_max_deal_prob"]
# for model in models:
#     y_pred = model.predict(test_x)
#     # oof_pred = oof_pred + y_pred
#     c_pred = np.where(y_pred > test_max_value, test_max_value, y_pred)
#     oof_pred = oof_pred + c_pred
# oof_pred = oof_pred / split_num
#
# submission = pd.DataFrame()
# submission["item_id"] = test["item_id"]
# submission["deal_probability"] = oof_pred.clip(0.0, 1.0)
# submission.to_csv(OUTPUT_PRED, index=False)
#
# print(train["deal_probability"].describe())
# logger.info(submission.describe())
# print(submission.describe())
# timer.time("done submission in ")
