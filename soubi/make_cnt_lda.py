import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "avito")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
SUBMISSION = os.path.join(APP_ROOT, "submission")
PRED_TRAIN = os.path.join(OUTPUT_DIR, "pred_train_next.csv")
PRED_TEST = os.path.join(OUTPUT_DIR, "pred_test_next.csv")
GAZOU_TRAIN = os.path.join(OUTPUT_DIR, "image_train.csv")
GAZOU_TEST = os.path.join(OUTPUT_DIR, "image_test.csv")
MODEL_FILE = os.path.join(SUBMISSION, "pred_model.txt")
from avito.common import filename_getter
LDA_DENSE_S_TRAIN, LDA_DESC_S_TRAIN, LDA_TITLE_S_TRAIN = filename_getter.get_lda_filename(OUTPUT_DIR, "stem", "train")
LDA_DENSE_S_TEST, LDA_DESC_S_TEST, LDA_TITLE_S_TEST = filename_getter.get_lda_filename(OUTPUT_DIR, "stem", "test")
LDA_DENSE_CNT_S_TRAIN, LDA_DESC_CNT_S_TRAIN, LDA_TITLE_CNT_S_TRAIN = \
    filename_getter.get_lda_filename(OUTPUT_DIR, "stem_cnt", "train")
LDA_DENSE_CNT_S_TEST, LDA_DESC_CNT_S_TEST, LDA_TITLE_CNT_S_TEST = \
    filename_getter.get_lda_filename(OUTPUT_DIR, "stem_cnt", "test")

DENSE_CNT15_COLS, DENSE_CNT15_TRAIN, DENSE_CNT15_TEST = \
    filename_getter.get_filename(OUTPUT_DIR, "stem_cnt15_dense", "cnt")
DESC_CNT15_COLS, DESC_CNT15_TRAIN, DESC_CNT15_TEST = \
    filename_getter.get_filename(OUTPUT_DIR, "stem_cnt15_desc", "cnt")
TITLE_CNT15_COLS, TITLE_CNT15_TRAIN, TITLE_CNT15_TEST = \
    filename_getter.get_filename(OUTPUT_DIR, "stem_cnt15_title", "cnt")
DESC_TF_COLS, DESC_TF_TRAIN, DESC_TF_TEST = filename_getter.get_filename(OUTPUT_DIR, "stem_desc", "tf")
TITLE_TF_COLS, TITLE_TF_TRAIN, TITLE_TF_TEST = filename_getter.get_filename(OUTPUT_DIR, "stem_title", "tf")
TITLE_CNT_COLS, TITLE_CNT_TRAIN, TITLE_CNT_TEST = filename_getter.get_filename(OUTPUT_DIR, "stem_title", "cnt")
DENSE_TF_COLS, DENSE_TF_TRAIN, DENSE_TF_TEST = filename_getter.get_filename(OUTPUT_DIR, "stem_title_desc", "tf")

import pandas as pd
import numpy as np
import scipy.sparse
import gc
from sklearn import model_selection
from sklearn.decomposition import LatentDirichletAllocation
from dask import dataframe as dd
from avito.common import csv_loader, column_selector, pocket_lgb, pocket_timer, pocket_logger, holdout_validator
from avito.fe import additional_fe
# import gensim

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
dtypes = csv_loader.get_featured_dtypes()
predict_col = column_selector.get_predict_col()
lgb_col = column_selector.get_stem_col()

train = dd.read_csv(PRED_TRAIN).compute()
desc_train = scipy.sparse.load_npz(DENSE_TF_TRAIN)
title_train = scipy.sparse.load_npz(TITLE_CNT_TRAIN)

test = dd.read_csv(PRED_TEST).compute()
desc_test = scipy.sparse.load_npz(DENSE_TF_TEST)
title_test = scipy.sparse.load_npz(TITLE_CNT_TEST)
timer.time("load csv in ")

print(title_train.shape)

# desc_train = desc_train[:2000]
# desc_test = desc_test[:1000]
print(desc_train.shape)
print(desc_test.shape)
merged = scipy.sparse.vstack([desc_train, desc_test])
print(merged.shape)
timer.time("start lda")

lda = LatentDirichletAllocation(n_components=10, learning_method="online", random_state=99)
topics = lda.fit_transform(merged)
timer.time("end lda")

print(topics)
print(topics.shape)

train_rows = desc_train.shape[0]
print(train_rows)
topic_train = topics[:train_rows]
topic_test = topics[train_rows:]
print(topic_train.shape)
print(topic_test.shape)
np.save(LDA_DENSE_S_TRAIN, topic_train)
np.save(LDA_DENSE_S_TEST, topic_test)
