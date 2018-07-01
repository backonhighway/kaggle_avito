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
OUTPUT_PRED = os.path.join(SUB_DIR, "strong1-1.csv")
OUTPUT_CV_PRED = os.path.join(SUB_DIR, "strong1-1_cv.csv")
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
from avito.common import csv_loader, no_user_columns, pocket_lgb, pocket_timer, pocket_logger, holdout_validator
from avito.fe import additional_fe

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
dtypes = csv_loader.get_featured_dtypes()
predict_col = no_user_columns.get_predict_col()
spain_col = ["pred_R", "pred_text_word", "pred_text_word_ridge", "pred_im",
             "pred_cat1", "pred_FT1", "pred_RNN1", "xgb1_tfidf", "pred_lgb2", "pred_price"]
predict_col.extend(spain_col)
files = [DENSE_TF_COLS, TITLE_CNT_COLS]
names = ["dense_tf", "title_cnt"]
lgb_col = no_user_columns.get_cols_from_files(files, names, predict_col)

spain_train, spain_test = csv_loader.load_spain()
cv_folds = pd.read_csv(CV_FOLDS)

train = dd.read_csv(PRED_TRAIN).compute()
gazou = dd.read_csv(GAZOU_TRAIN).compute()
gazou["image"] = gazou["image"].apply(lambda w: w.replace(".jpg", ""))
train = pd.merge(train, gazou, on="image", how="left")
train = pd.merge(train, cv_folds, on="item_id", how="left")
train = pd.merge(train, spain_train, on="item_id", how="left")

test = dd.read_csv(PRED_TEST).compute()
gazou = dd.read_csv(GAZOU_TEST).compute()
gazou["image"] = gazou["image"].apply(lambda w: w.replace(".jpg", ""))
test = pd.merge(test, gazou, on="image", how="left")
test = pd.merge(test, spain_test, on="item_id", how="left")

dense_tf_train = scipy.sparse.load_npz(DENSE_TF_TRAIN)
title_cnt_train = scipy.sparse.load_npz(TITLE_CNT_TRAIN)

dense_tf_test = scipy.sparse.load_npz(DENSE_TF_TEST)
title_cnt_test = scipy.sparse.load_npz(TITLE_CNT_TEST)
timer.time("load csv in ")

train_x = train[predict_col]
train_stack = [scipy.sparse.csr_matrix(train_x), dense_tf_train, title_cnt_train]
train_x = scipy.sparse.hstack(train_stack).tocsr()
train_y = train["deal_probability"]

test_x = test[predict_col]
test_stack = [scipy.sparse.csr_matrix(test_x), dense_tf_test, title_cnt_test]
test_x = scipy.sparse.hstack(test_stack).tocsr()

submission = pd.DataFrame()
submission["item_id"] = test["item_id"]
submission["deal_probability"] = 0

train_final_out = pd.DataFrame()
train_final_out["item_id"] = train["item_id"]
train_final_out["cv_pred"] = 0
timer.time("prepare train in ")

bagging_num = 2
for bagging_index in range(bagging_num):
    total_score = 0
    models = []
    train_preds = []
    seed = 99 * bagging_index
    lgb = pocket_lgb.get_simple_lgb(seed)
    for split_index in range(1, 5):
        short_timer = pocket_timer.GoldenTimer(logger)
        mask = train["fold"] != split_index
        train_index = train[mask].index.values
        valid_index = train[~mask].index.values
        train_x_, valid_x_ = train_x[train_index], train_x[valid_index]
        train_y_, valid_y_ = train_y.iloc[train_index], train_y.iloc[valid_index]

        model = lgb.do_train_avito(train_x_, valid_x_, train_y_, valid_y_, lgb_col)
        score = model.best_score["valid_0"]["rmse"]
        total_score += score
        y_pred = model.predict(test_x)
        train_reverse_pred = model.predict(valid_x_)
        models.append(model)

        submission["deal_probability"] = submission["deal_probability"] + y_pred
        train_cv_prediction = pd.DataFrame()
        valid_ = train.iloc[valid_index]
        train_cv_prediction["item_id"] = valid_["item_id"]
        train_cv_prediction["cv_pred"] = train_reverse_pred
        train_preds.append(train_cv_prediction)
        short_timer.time("done one set in")

    train_output = pd.concat(train_preds, axis=0)
    train_output["cv_pred"] = np.clip(train_output["cv_pred"], 0.0, 1.0)
    train_final_out["cv_pred"] = train_final_out["cv_pred"] + train_output["cv_pred"]

    lgb.show_feature_importance(models[0])
    avg_score = str(total_score / 4)
    print("average score= " + avg_score)
    logger.info("average score= " + avg_score)
    timer.time("end train in ")


submission["deal_probability"] = submission["deal_probability"] / (4 * bagging_num)
submission["deal_probability"] = np.clip(submission["deal_probability"], 0.0, 1.0)
submission.to_csv(OUTPUT_PRED, index=False)

train_final_out["cv_pred"] = train_final_out["cv_pred"] / bagging_num
train_final_out["cv_pred"] = np.clip(train_final_out["cv_pred"], 0.0, 1.0)
train_final_out.to_csv(OUTPUT_CV_PRED, index=False)

print(train["deal_probability"].describe())
logger.info(train_final_out.describe())
logger.info(submission.describe())
print(train_final_out.describe())
print(submission.describe())
timer.time("done submission in ")

