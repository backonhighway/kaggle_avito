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
PSEUDO_PRED_99 = os.path.join(SUB_DIR, "submission99p.csv")
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
BEST_SUB = os.path.join(SUB_DIR, "submission.csv")
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

test = pd.merge(test, best_sub, on="item_id", how="left")
train_y = train["deal_probability"]
test_y = test["deal_probability"]
train_x = train[predict_col]
test_x = test[predict_col]

the_stack = [scipy.sparse.csr_matrix(train_x), dense_tf_train, title_cnt_train]
train_x = scipy.sparse.hstack(the_stack)
print(train_x.shape)
test_stack = [scipy.sparse.csr_matrix(test_x), dense_tf_test, title_cnt_test]
test_x = scipy.sparse.hstack(test_stack)
print(test_x.shape)

X_train, X_valid, y_train, y_valid = \
    model_selection.train_test_split(train_x, train_y, test_size=0.2, random_state=99)

X_train = scipy.sparse.vstack([X_train, test_x])
y_train = pd.concat([y_train, test_y], axis=0)
print(X_train.shape)
print(y_train.shape)

timer.time("prepare train in ")
lgb = pocket_lgb.GoldenLgb()
model = lgb.do_train_avito(X_train, X_valid, y_train, y_valid, lgb_col)
lgb.show_feature_importance(model)
#exit(0)

timer.time("end train in ")
# validator = holdout_validator.HoldoutValidator(model, X_valid, y_valid, max_prob)
# validator.validate()
# validator.output_prediction(PREDICTION)


test_x = test[predict_col]
y_pred = model.predict(test_x)
y_pred = np.clip(y_pred, 0.0, 1.0)
submission = pd.DataFrame()
submission["item_id"] = test["item_id"]
submission["deal_probability"] = y_pred
submission.to_csv(PSEUDO_PRED_99, index=False)

print(train["deal_probability"].describe())
logger.info(submission.describe())
print(submission.describe())
timer.time("done submission in ")