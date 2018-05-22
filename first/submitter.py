import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "avito")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
PRED_TRAIN = os.path.join(OUTPUT_DIR, "pred_train.csv")
PRED_TEST = os.path.join(OUTPUT_DIR, "pred_test.csv")
OUTPUT_PRED = os.path.join(OUTPUT_DIR, "submission.csv")
MODEL_FILE = os.path.join(OUTPUT_DIR, "pred_model.txt")

import pandas as pd
import numpy as np
import gc
from sklearn import model_selection
from dask import dataframe as dd
from avito.common import csv_loader, column_selector, holdout_validator, pocket_lgb, pocket_timer, pocket_logger

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
dtypes = csv_loader.get_featured_dtypes()
predict_col = column_selector.get_predict_col()

train = dd.read_csv(PRED_TRAIN).compute()
test = dd.read_csv(PRED_TEST).compute()
timer.time("load csv in ")

train_y = train["deal_probability"]
train_x = train[predict_col]
X_train, X_valid, y_train, y_valid = model_selection.train_test_split(train_x, train_y, test_size=0.2, random_state=99)

timer.time("prepare train in ")
lgb = pocket_lgb.GoldenLgb()
model = lgb.do_train_sk(X_train, X_valid, y_train, y_valid)
lgb.show_feature_importance(model)
model.save_model(MODEL_FILE)
del train, X_train, X_valid, y_train, y_valid
gc.collect()
timer.time("end train in ")


y_pred = model.predict(test[predict_col])
y_pred = np.clip(y_pred, 0.0, 1.0)
submission = pd.DataFrame()
submission["item_id"] = test["item_id"]
submission["deal_probability"] = y_pred
submission.to_csv(OUTPUT_PRED, index=False)
logger.info(submission.describe())
print(submission.describe())
timer.time("done submission in ")
