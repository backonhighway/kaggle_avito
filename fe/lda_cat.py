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
CAT_LDA_TRAIN = os.path.join(OUTPUT_DIR, "cat_lda_train.csv")
CAT_LDA_TEST = os.path.join(OUTPUT_DIR, "cat_lda_test.csv")
import numpy as np
import pandas as pd
from avito.common import pocket_lda, pocket_timer, pocket_logger
from sklearn import preprocessing

logger = pocket_logger.get_my_logger()
whole_timer = pocket_timer.GoldenTimer(logger)
timer = pocket_timer.GoldenTimer(logger)
train = pd.read_csv(PRED_TRAIN)
test = pd.read_csv(PRED_TEST)
timer.time("load csv")

le = preprocessing.LabelEncoder()
col = "user_id"
le.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
train[col] = le.transform(train[col].values.astype('str'))
test[col] = le.transform(test[col].values.astype('str'))

cols = ['user_id', 'city', 'image_top_1', 'param_all']
for col in cols:
    train[col].fillna(0, inplace=True)
    test[col].fillna(0, inplace=True)
    train[col] = train[col].astype(int)
    test[col] = test[col].astype(int)

maker = pocket_lda.GoldenLDA(timer)
lda_train, lda_test = maker.create_features(train, test)

logger.info(list(lda_train))

train = pd.concat([train, lda_train], axis=1)
print(train.shape)

train.to_csv(CAT_LDA_TRAIN, index=False)
test.to_csv(CAT_LDA_TEST, index=False)

whole_timer.time("all done")


