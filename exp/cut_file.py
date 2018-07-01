import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "avito")
INPUT_DIR = os.path.join(APP_ROOT, "input")
POCKET_DIR = os.path.join(APP_ROOT, "pocket", "others")
PRED_TRAIN = os.path.join(POCKET_DIR, "pred_train_all.csv")
PRED_TEST = os.path.join(POCKET_DIR, "pred_test_all.csv")
OUTPUT_LDA_TRAIN = os.path.join(POCKET_DIR, "lda_train_short.csv")
OUTPUT_LDA_TEST = os.path.join(POCKET_DIR, "lda_test_short.csv")
import pandas as pd
from avito.common import column_selector

train = pd.read_csv(PRED_TRAIN)
test = pd.read_csv(PRED_TEST)

lda_col = column_selector.get_lda_col(False)
lda_col_all = column_selector.get_lda_col(True)

print(lda_col)
use_col = ["item_id"]
use_col.extend(lda_col)
#use_col.extend(lda_col_all)

train = train[use_col]
test = test[use_col]

print("output to csv...")
train.to_csv(OUTPUT_LDA_TRAIN, index=False)
test.to_csv(OUTPUT_LDA_TEST, index=False)
