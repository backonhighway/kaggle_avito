import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "avito")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
SUBMISSION = os.path.join(APP_ROOT, "submission")
PRED_TRAIN = os.path.join(OUTPUT_DIR, "pred_train.csv")
PRED_TEST = os.path.join(OUTPUT_DIR, "pred_test.csv")
OUTPUT_PRED = os.path.join(SUBMISSION, "submission.csv")
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
from avito.common import csv_loader, column_selector, pocket_lgb, pocket_timer, pocket_logger

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
dtypes = csv_loader.get_featured_dtypes()
predict_col = column_selector.get_predict_col()
lgb_col = column_selector.get_pred_tf_col()

train = dd.read_csv(PRED_TRAIN).compute()
desc_train = scipy.sparse.load_npz(DENSE_TF_TRAIN)
title_train = scipy.sparse.load_npz(TITLE_CNT_TRAIN)

test = dd.read_csv(PRED_TEST).compute()
desc_test = scipy.sparse.load_npz(DENSE_TF_TEST)
title_test = scipy.sparse.load_npz(TITLE_CNT_TEST)
timer.time("load csv in ")

train = train[predict_col]
train = scipy.sparse.hstack([scipy.sparse.csr_matrix(train), desc_train, title_train])
test = test[predict_col]
test = scipy.sparse.hstack([scipy.sparse.csr_matrix(test), desc_test, title_test])

# train_y = train["deal_probability"]
# train_x = train[predict_col]
# train_x = scipy.sparse.hstack([scipy.sparse.csr_matrix(train_x), desc_train, title_train])

split_num = 5
skf = model_selection.KFold(n_splits=split_num)
lgb = pocket_lgb.GoldenLgb()
total_score = 0
models = []
for train_index, test_index in skf.split(train):
    # train_np = train.iloc[train_index]
    # test_np = train.iloc[test_index]
    # train_df = pd.DataFrame(train_np)
    # test_df = pd.DataFrame(test_np)
    train_df = train.ix[train_index]
    test_df = train.ix[test_index]

    model = lgb.do_train(train_df, test_df, lgb_col)
    score = model.best_score["valid_0"]["rmse"]
    total_score += score

    models.append(model)

print("average score= ", total_score / split_num)
lgb.show_feature_importance(models[0])
timer.time("end train in ")

train_rows = train.shape[0]
test_rows = test.shape[0]
oof_train = np.zeros((train_rows,))
oof_pred = np.zeros((test_rows,))
for model in models:
    y_pred = model.predict(test)
    max_value = test["parent_max_deal_prob"]
    c_pred = np.where(y_pred > max_value, max_value, y_pred)
    oof_pred = oof_pred + c_pred
oof_pred = oof_pred / split_num

submission = pd.DataFrame()
submission["item_id"] = test["item_id"]
submission["deal_probability"] = oof_pred
submission.to_csv(OUTPUT_PRED, index=False)
logger.info(submission.describe())
print(submission.describe())
timer.time("done submission in ")
