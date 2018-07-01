import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "avito")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
SPAIN_DIR = os.path.join(APP_ROOT, "spain")
POCKET_DIR = os.path.join(APP_ROOT, "pocket", "strong")
PRED_TRAIN = os.path.join(OUTPUT_DIR, "pred_train.csv")
CV_FOLDS = os.path.join(SPAIN_DIR, "train_folds.csv")
SUB_DIR = os.path.join(APP_ROOT, "submission")
OUTPUT_PRED = os.path.join(SUB_DIR, "strong2.csv")
OUTPUT_CV_PRED = os.path.join(SUB_DIR, "strong2_cv.csv")
import pandas as pd
from sklearn import metrics

train = pd.read_csv(PRED_TRAIN)
use_col = ["item_id", "deal_probability"]
train = train[use_col]

dfs = []
filenames = [OUTPUT_CV_PRED]
for filename in filenames:
    df = pd.read_csv(filename)
    df = pd.merge(df, train, on="item_id", how="left")
    dfs.append(df)

for some_df in dfs:
    y_pred = some_df["cv_pred"]
    y_true = some_df["deal_probability"]
    score = metrics.mean_squared_error(y_true, y_pred) ** 0.5
    print(score)




