import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "avito")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
SPAIN_DIR = os.path.join(APP_ROOT, "spain")
SUB_DIR = os.path.join(APP_ROOT, "submission")
PRED_TRAIN = os.path.join(OUTPUT_DIR, "pred_train.csv")
PRED_TEST = os.path.join(OUTPUT_DIR, "pred_test.csv")
GAZOU_TRAIN = os.path.join(OUTPUT_DIR, "image_train.csv")
GAZOU_TEST = os.path.join(OUTPUT_DIR, "image_test.csv")
CV_FOLDS = os.path.join(SPAIN_DIR, "train_folds.csv")
OUTPUT_PRED = os.path.join(SUB_DIR, "pl_simple3.csv")
OUTPUT_CV_PRED = os.path.join(SUB_DIR, "pl_simple3_cv.csv")
import pandas as pd
import numpy as np
import scipy.sparse
import gc
from sklearn import model_selection
from dask import dataframe as dd
from avito.common import csv_loader, no_user_columns, pocket_lgb, pocket_timer, pocket_logger, holdout_validator
from avito.fe import additional_fe
from sklearn import metrics

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
dtypes = csv_loader.get_featured_dtypes()
predict_col = no_user_columns.get_predict_col()

cv_folds = pd.read_csv(CV_FOLDS)
train = dd.read_csv(PRED_TRAIN).compute()
gazou = dd.read_csv(GAZOU_TRAIN).compute()
gazou["image"] = gazou["image"].apply(lambda w: w.replace(".jpg", ""))
train = pd.merge(train, gazou, on="image", how="left")
train = pd.merge(train, cv_folds, on="item_id", how="left")

test = train[train["fold"] == 1].copy()
train = train[train["fold"] != 1]
timer.time("load csv in ")

test_x = test[predict_col]

submission = pd.DataFrame()
submission["item_id"] = test["item_id"]
submission["true_dp"] = test["deal_probability"]
submission["deal_probability"] = 0


bagging_num = 2
for bagging_index in range(bagging_num):
    total_score = 0
    models = []
    train_preds = []
    seed = 99 * bagging_index
    lgb = pocket_lgb.get_simple_lgb(seed)
    for split_index in range(2, 5):
        short_timer = pocket_timer.GoldenTimer(logger)
        mask = train["fold"] != split_index
        train_ = train[mask]
        valid_ = train[~mask]
        train_x, train_y = train_[predict_col], train_["deal_probability"]
        valid_x, valid_y = valid_[predict_col], valid_["deal_probability"]

        model = lgb.do_train_avito(train_x, valid_x, train_y, valid_y, predict_col)
        score = model.best_score["valid_0"]["rmse"]
        total_score += score
        y_pred = model.predict(test_x)
        y_true = test["deal_probability"]
        round1_score = metrics.mean_squared_error(y_true, y_pred) ** 0.5
        print(round1_score)
        print("round1---------------------------")

        train_x_temp = pd.concat([train_x, test_x])
        pred_series = pd.Series(y_pred)
        train_y_temp = pd.concat([train_y, pred_series])

        model = lgb.do_train_avito(train_x_temp, valid_x, train_y_temp, valid_y, predict_col)
        score = model.best_score["valid_0"]["rmse"]
        total_score += score
        y_pred = model.predict(test_x)
        round2_score = metrics.mean_squared_error(y_true, y_pred) ** 0.5
        print(round2_score)
        print("round2---------------------------")

        models.append(model)

        submission["deal_probability"] = submission["deal_probability"] + y_pred
        short_timer.time("done one set in")



    lgb.show_feature_importance(models[0])
    avg_score = str(total_score / 3)
    print("average score= " + avg_score)
    logger.info("average score= " + avg_score)
    timer.time("end train in ")

submission["deal_probability"] = submission["deal_probability"] / (3 * bagging_num)
submission["deal_probability"] = np.clip(submission["deal_probability"], 0.0, 1.0)
print(submission.describe())

y_true = submission["true_dp"]
y_pred = submission["deal_probability"]
round1_score = metrics.mean_squared_error(y_true, y_pred) ** 0.5
print(round1_score)
print("round1---------------------------")
