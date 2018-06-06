import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "avito")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
PRED_TRAIN = os.path.join(OUTPUT_DIR, "pred_train.csv")
PRED_TEST = os.path.join(OUTPUT_DIR, "pred_test.csv")
import pandas as pd


train = pd.read_csv(PRED_TRAIN)
test = pd.read_csv(PRED_TEST)
drop_col = ["title", "description"]
train.drop(drop_col, axis=1, inplace=True)
test.drop(drop_col, axis=1, inplace=True)

train.to_csv(PRED_TRAIN, index=False)
test.to_csv(PRED_TEST, index=False)
