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
BEST_SUB = os.path.join(SUB_DIR, "lb_2198.csv")
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
# files = [DENSE_TF_COLS, DESC_TF_COLS, TITLE_TF_COLS, DENSE_CNT15_COLS, DESC_CNT15_COLS, TITLE_CNT15_COLS]
# names = ["dense_tf_", "desc_tf_", "title_tf", "dense_cnt15_", "desc_cnt15_", "title_cnt15_"]
files = [DENSE_TF_COLS, TITLE_CNT15_COLS]
names = ["dense_tf_", "title_cnt15_"]
lgb_col = column_selector.get_cols_from_files(files, names)

train = dd.read_csv(PRED_TRAIN).compute()
gazou = dd.read_csv(GAZOU_TRAIN).compute()
train = pd.merge(train, gazou, on="image", how="left")
test = dd.read_csv(PRED_TEST).compute()
gazou = dd.read_csv(GAZOU_TEST).compute()
gazou["image"] = gazou["image"].apply(lambda w: w.replace(".jpg", ""))
test = pd.merge(test, gazou, on="image", how="left")

dense_tf_train = scipy.sparse.load_npz(DENSE_TF_TRAIN)
title_tf_train = scipy.sparse.load_npz(TITLE_TF_TRAIN)
title_cnt_train = scipy.sparse.load_npz(TITLE_CNT15_TRAIN)
title_cnt_all_train = scipy.sparse.load_npz(TITLE_CNT_TRAIN)
dense_tf_test = scipy.sparse.load_npz(DENSE_TF_TEST)
title_cnt_test = scipy.sparse.load_npz(TITLE_CNT15_TEST)

best_sub = pd.read_csv(BEST_SUB)
timer.time("load csv in ")

p_test = pd.merge(test, best_sub, on="item_id", how="left")
train_y = train["deal_probability"]
p_test_y = p_test["deal_probability"]
train_x = train[predict_col]
p_test_x = p_test[predict_col]

train_x = scipy.sparse.hstack([scipy.sparse.csr_matrix(train_x), dense_tf_train, title_cnt_train]).tocsr()
p_test_x = scipy.sparse.hstack([scipy.sparse.csr_matrix(p_test_x), dense_tf_test, title_cnt_test]).tocsr()
print(train_x.shape)
print(p_test_x.shape)

# train_x = scipy.sparse.vstack([train_x, p_test_x])
# train_y = pd.concat([train_y, p_test_y], axis=0)
# print(train_x.shape)
# print(train_y.shape)
timer.time("prepare train in ")

split_num = 4
skf = model_selection.KFold(n_splits=split_num, shuffle=False)
lgb = pocket_lgb.GoldenLgb()
total_score = 0
models = []
subs = []
for (train_index, test_index), (ptrain_idx, ptest_idx) in zip(skf.split(train), skf.split(p_test)):
    X_train, X_test = train_x[train_index], train_x[test_index]
    y_train, y_test = train_y.iloc[train_index], train_y.iloc[test_index]

    pX_train, pX_test = p_test_x[ptrain_idx], p_test_x[ptest_idx]
    py_train, py_test = p_test_y.iloc[ptrain_idx], p_test_y.iloc[ptest_idx]

    X_train = scipy.sparse.vstack([X_train, pX_train])
    y_train = pd.concat([y_train, py_train], axis=0)

    model = lgb.do_train_avito(X_train, X_test, y_train, y_test, lgb_col)
    score = model.best_score["valid_0"]["rmse"]
    total_score += score
    models.append(model)

    p_test_id = p_test.iloc[ptest_idx]
    y_pred = model.predict(pX_test)
    submission = pd.DataFrame()
    submission["item_id"] = p_test_id["item_id"]
    submission["deal_probability"] = y_pred
    subs.append(submission)
    timer.time("done one set in")

lgb.show_feature_importance(models[0])
print("average score= ", total_score / split_num)
timer.time("end train in ")

submission = pd.concat(subs, axis=0)
submission["deal_probability"] = np.clip(submission["deal_probability"], 0.0, 1.0)
submission.to_csv(OUTPUT_PRED, index=False)

print(train["deal_probability"].describe())
logger.info(submission.describe())
print(submission.describe())
timer.time("done submission in ")
