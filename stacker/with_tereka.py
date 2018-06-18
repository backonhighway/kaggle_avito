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
tereka_col = ["tereka1", "tereka2", "tereka3", "pocket1", "pocket2", "pocket3"]
predict_col.extend(tereka_col)

train = dd.read_csv(PRED_TRAIN).compute()
gazou = dd.read_csv(GAZOU_TRAIN).compute()
train = pd.merge(train, gazou, on="image", how="left")
test = dd.read_csv(PRED_TEST).compute()
gazou = dd.read_csv(GAZOU_TEST).compute()
gazou["image"] = gazou["image"].apply(lambda w: w.replace(".jpg", ""))
test = pd.merge(test, gazou, on="image", how="left")
timer.time("load csv in ")

tereka_train, tereka_test = csv_loader.load_tereka()
train = pd.merge(train, tereka_train, on="item_id", how="left")
test = pd.merge(test, tereka_test, on="item_id", how="left")
pocket_train, pocket_test = csv_loader.load_pocket()
train = pd.merge(train, pocket_train, on="item_id", how="left")
test = pd.merge(test, pocket_test, on="item_id", how="left")
timer.time("load stacking csv in ")

train_y = train["deal_probability"]
train_x = train[predict_col]
test_x = test[predict_col]
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
    random_state = 71 * bagging_index
    skf = model_selection.KFold(n_splits=split_num, shuffle=True, random_state=random_state)
    lgb = pocket_lgb.get_stacking_lgb(random_state)
    total_score = 0
    models = []
    train_preds = []
    for train_index, test_index in skf.split(train):
        X_train, X_test = train_x.iloc[train_index], train_x.iloc[test_index]
        y_train, y_test = train_y.iloc[train_index], train_y.iloc[test_index]

        model = lgb.do_train_avito(X_train, X_test, y_train, y_test, predict_col)
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
    avg_score = str(total_score / split_num)
    print("average score= " + avg_score )
    logger.info("average score= " + avg_score)
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

