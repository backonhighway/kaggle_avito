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
DESC_TF_COLS = os.path.join(OUTPUT_DIR, "desc_tf_col.csv")
DESC_TF_TRAIN = os.path.join(OUTPUT_DIR, "desc_tf_train.npz")
DESC_TF_TEST = os.path.join(OUTPUT_DIR, "desc_tf_test.npz")
TITLE_TF_COLS = os.path.join(OUTPUT_DIR, "title_tf_col.csv")
TITLE_TF_TRAIN = os.path.join(OUTPUT_DIR, "title_tf_train.npz")
TITLE_TF_TEST = os.path.join(OUTPUT_DIR, "title_tf_test.npz")

import numpy as np
import pandas as pd
import scipy.sparse
import gc
from avito.common import pocket_timer, pocket_logger, column_selector
from avito.fe import big_bow

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
#dtypes = csv_loader.get_featured_dtypes()
#predict_col = column_selector.get_predict_col()

# train = pd.read_csv(ORG_TRAIN, nrows=1000*10)
# test = pd.read_csv(ORG_TEST, nrows=1000*10)
train = pd.read_csv(ORG_TRAIN)
test = pd.read_csv(ORG_TEST)
timer.time("read csv")
print("-"*40)

desc_train, desc_test, desc_tf_names = big_bow.make_desc_tf(train, test)
timer.time("done desc tf-idf")
temp_df = pd.DataFrame(desc_tf_names)
temp_df.to_csv(DESC_TF_COLS, index=False, header=None)
scipy.sparse.save_npz(DESC_TF_TRAIN, desc_train)
scipy.sparse.save_npz(DESC_TF_TEST, desc_test)
timer.time("saved desc tf-idf")

title_train, title_test, title_tf_names = big_bow.make_title_tf(train, test)
timer.time("done title tf-idf")
temp_df = pd.DataFrame(title_tf_names)
temp_df.to_csv(TITLE_TF_COLS, index=False, header=None)
scipy.sparse.save_npz(TITLE_TF_TRAIN, title_train)
scipy.sparse.save_npz(TITLE_TF_TEST, title_test)
timer.time("saved title tf-idf")

