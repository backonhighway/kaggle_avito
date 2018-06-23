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
ACTIVE_TRAIN = os.path.join(INPUT_DIR, "train_active.csv")
ACTIVE_TEST = os.path.join(INPUT_DIR, "test_active.csv")
CAT_LDA_TRAIN = os.path.join(OUTPUT_DIR, "cat_lda_all_train.csv")
CAT_LDA_TEST = os.path.join(OUTPUT_DIR, "cat_lda_all_test.csv")
import numpy as np
import pandas as pd
from avito.common import pocket_lda, pocket_timer, pocket_logger
from sklearn import preprocessing

logger = pocket_logger.get_my_logger()
whole_timer = pocket_timer.GoldenTimer(logger)
timer = pocket_timer.GoldenTimer(logger)
train = pd.read_csv(PRED_TRAIN)
test = pd.read_csv(PRED_TEST)
train_active = pd.read_csv(ACTIVE_TRAIN)
test_active = pd.read_csv(ACTIVE_TEST)
timer.time("load csv")

def do_prep(train, test, train_active, test_active):
    train = get_param_all(train)
    test = get_param_all(test)
    train_active = get_param_all(train_active)
    test_active = get_param_all(test_active)

    # "item_id", "user_id", title, description
    cat_cols = ["region", "city", "parent_category_name", "category_name",
                "param_1", "param_2", "param_3", "user_type", "param_all"]
    for col in cat_cols:
        print(col)
        le = preprocessing.LabelEncoder()
        # le.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
        le.fit(
            list(train[col].values.astype('str')) + list(test[col].values.astype('str')) +
            list(train_active[col].values.astype('str')) + list(test_active[col].values.astype('str'))
        )
        train[col] = le.transform(train[col].values.astype('str'))
        test[col] = le.transform(test[col].values.astype('str'))
        train_active[col] = le.transform(train_active[col].values.astype('str'))
        test_active[col] = le.transform(test_active[col].values.astype('str'))

    all_df = pd.concat([train, test, train_active, test_active])
    #all_periods = pd.concat([train_period, test_period])
    all_train = pd.concat([train, train_active])
    all_test = pd.concat([test, test_active])
    #return train, test, train_active, test_active, all_df, all_periods, all_train, all_test


def get_param_all(df):
    param_list = ["param_1", "param_2", "param_3"]
    for some_param in param_list:
        df[some_param].fillna("_", inplace=True)
    df['param_all'] = df["param_1"] + df["param_2"] + df["param_3"]
    return df



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


