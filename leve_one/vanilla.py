import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "avito")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
SUB_DIR = os.path.join(APP_ROOT, "submission")
PRED_TRAIN = os.path.join(OUTPUT_DIR, "pred_train.csv")
PRED_TEST = os.path.join(OUTPUT_DIR, "pred_test.csv")
GAZOU_TRAIN = os.path.join(OUTPUT_DIR, "image_train.csv")
GAZOU_TEST = os.path.join(OUTPUT_DIR, "image_test.csv")
OUTPUT_PRED = os.path.join(SUB_DIR, "submission.csv")
OUTPUT_CV_PRED = os.path.join(SUB_DIR, "cv_pred.csv")
from avito.common import filename_getter
DENSE_CNT15_COLS, DENSE_CNT15_TRAIN, DENSE_CNT15_TEST = \
    filename_getter.get_filename(OUTPUT_DIR, "vanilla_cnt15_dense", "cnt")
DESC_CNT15_COLS, DESC_CNT15_TRAIN, DESC_CNT15_TEST = \
    filename_getter.get_filename(OUTPUT_DIR, "vanilla_cnt15_desc", "cnt")
TITLE_CNT15_COLS, TITLE_CNT15_TRAIN, TITLE_CNT15_TEST = \
    filename_getter.get_filename(OUTPUT_DIR, "vanilla_cnt15_title", "cnt")
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
files = [DENSE_CNT15_COLS, DESC_CNT15_COLS, TITLE_CNT15_COLS,
         VDENSE_TF_COLS, VDESC_TF_COLS, VTITLE_TF_COLS,]
names = ["dense_cnt15_", "desc_cnt15_", "title_cnt15_",
         "vdense_tf_", "vdesc_tf_", "vtitle_tf",]
lgb_col = column_selector.get_cols_from_files(files, names)

train = dd.read_csv(PRED_TRAIN).compute()
gazou = dd.read_csv(GAZOU_TRAIN).compute()
train = pd.merge(train, gazou, on="image", how="left")
test = dd.read_csv(PRED_TEST).compute()
gazou = dd.read_csv(GAZOU_TEST).compute()
gazou["image"] = gazou["image"].apply(lambda w: w.replace(".jpg", ""))
test = pd.merge(test, gazou, on="image", how="left")

dense_cnt_train = scipy.sparse.load_npz(DENSE_CNT15_TRAIN)
desc_cnt_train = scipy.sparse.load_npz(DESC_CNT15_TRAIN)
title_cnt_train = scipy.sparse.load_npz(TITLE_CNT15_TRAIN)
vdense_tf_train = scipy.sparse.load_npz(VDENSE_TF_TRAIN)
vdesc_tf_train = scipy.sparse.load_npz(VDESC_TF_TRAIN)
vtitle_tf_train = scipy.sparse.load_npz(VTITLE_TF_TRAIN)

dense_cnt_test = scipy.sparse.load_npz(DENSE_CNT15_TEST)
desc_cnt_test = scipy.sparse.load_npz(DESC_CNT15_TEST)
title_cnt_test = scipy.sparse.load_npz(TITLE_CNT15_TEST)
vdense_tf_test = scipy.sparse.load_npz(VDENSE_TF_TEST)
vdesc_tf_test = scipy.sparse.load_npz(VDESC_TF_TEST)
vtitle_tf_test = scipy.sparse.load_npz(VTITLE_TF_TEST)
timer.time("load csv in ")

train_y = train["deal_probability"]
train_x = train[predict_col]
train_stack = [scipy.sparse.csr_matrix(train_x),
               dense_cnt_train, desc_cnt_train, title_cnt_train, vdense_tf_train, vdesc_tf_train, vtitle_tf_train,]
train_x = scipy.sparse.hstack(train_stack).tocsr()

test_x = test[predict_col]
test_stack = [scipy.sparse.csr_matrix(test_x),
              dense_cnt_test, desc_cnt_test, title_cnt_test, vdense_tf_test, vdesc_tf_test, vtitle_tf_test,]
test_x = scipy.sparse.hstack(test_stack).tocsr()
timer.time("prepare train in ")

submission = pd.DataFrame()
submission["item_id"] = test["item_id"]
submission["deal_probability"] = 0

train_final_out = pd.DataFrame()
train_final_out["item_id"] = train["item_id"]
train_final_out["cv_pred"] = 0

bagging_num = 2
split_num = 5
for bagging_index in range(bagging_num):
    skf = model_selection.KFold(n_splits=split_num, shuffle=True, random_state=71 * bagging_index)
    lgb = pocket_lgb.GoldenLgb()
    total_score = 0
    models = []
    train_preds = []
    for train_index, test_index in skf.split(train):
        X_train, X_test = train_x[train_index], train_x[test_index]
        y_train, y_test = train_y.iloc[train_index], train_y.iloc[test_index]

        model = lgb.do_train_avito(X_train, X_test, y_train, y_test, lgb_col)
        score = model.best_score["valid_0"]["rmse"]
        total_score += score
        y_pred = model.predict(test_x)
        train_reverse_pred = model.predict(X_test)
        models.append(model)

        submission["deal_probability"] = submission["deal_probability"] + y_pred
        train_id = train.iloc[test_index]
        train_cv_prediction = pd.DataFrame()
        train_cv_prediction["item_id"] = train_id["item_id"]
        train_cv_prediction["cv_pred"] = train_reverse_pred
        train_preds.append(train_cv_prediction)
        timer.time("done one set in")

    train_output = pd.concat(train_preds, axis=0)
    train_output["cv_pred"] = np.clip(train_output["cv_pred"], 0.0, 1.0)
    train_final_out["cv_pred"] = train_final_out["cv_pred"] + train_output["cv_pred"]

    lgb.show_feature_importance(models[0])
    print("average score= ", total_score / split_num)
    timer.time("end train in ")


submission["deal_probability"] = submission["deal_probability"] / (bagging_num * split_num)
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

