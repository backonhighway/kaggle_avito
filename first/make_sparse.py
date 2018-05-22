import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "avito")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
ORG_TRAIN = os.path.join(INPUT_DIR, "train.csv")
ORG_TEST = os.path.join(INPUT_DIR, "test.csv")
PRED_TRAIN = os.path.join(OUTPUT_DIR, "pred_train.csv")
PRED_TEST = os.path.join(OUTPUT_DIR, "pred_test.csv")
TF_COLS = os.path.join(OUTPUT_DIR, "tf_col.csv")
TF_TRAIN = os.path.join(OUTPUT_DIR, "tf_train.npz")
TF_TEST = os.path.join(OUTPUT_DIR, "tf_test.npz")

import numpy as np
import pandas as pd
import scipy.sparse
import gc
from sklearn import preprocessing
from avito.common import pocket_timer, pocket_logger, column_selector
from avito.fe import sparse_bow

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
#dtypes = csv_loader.get_featured_dtypes()
#predict_col = column_selector.get_predict_col()

train = pd.read_csv(ORG_TRAIN)
test = pd.read_csv(ORG_TEST)
timer.time("read csv")
print("-"*40)

ret_train, ret_test, tf_names = sparse_bow.make_tfidf(train, test)
timer.time("done tf-idf")

temp_df = pd.DataFrame(tf_names)
temp_df.to_csv(TF_COLS, index=False, header=None)

scipy.sparse.save_npz(TF_TRAIN, ret_train)
scipy.sparse.save_npz(TF_TEST, ret_test)
