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

import numpy as np
import pandas as pd
import gc
from sklearn import preprocessing
from avito.common import pocket_timer, pocket_logger, column_selector
from avito.fe import do_fe

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
#dtypes = csv_loader.get_featured_dtypes()
#predict_col = column_selector.get_predict_col()

train = pd.read_csv(ORG_TRAIN)
test = pd.read_csv(ORG_TEST)
print(train.describe())
timer.time("read csv")
print("-"*40)

for some_df in (test, train):
    some_df["activation_date"] = pd.to_datetime(some_df["activation_date"])
    some_df["activation_dow"] = some_df["activation_date"].dt.dayofweek
    some_df["activation_day"] = some_df["activation_date"].dt.day

    #some_df["no_price"] = np.where(some_df["price"].isnull(), 1, 0)

    param_list = ["param_1", "param_2", "param_3"]
    for some_param in param_list:
        some_df[some_param].fillna("_", inplace=True)
    some_df['param_all'] = some_df["param_1"] + some_df["param_2"] + some_df["param_3"]

# "item_id", "user_id", title, description
cat_cols = ["region", "city", "parent_category_name", "category_name",
            "param_1", "param_2", "param_3", "user_type", "param_all"]
for col in cat_cols:
    print(col)
    le = preprocessing.LabelEncoder()
    #train[col] = le.fit_transform(train[col].astype("str"))
    le.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
    train[col] = le.transform(train[col].values.astype('str'))
    test[col] = le.transform(test[col].values.astype('str'))



print(train.info())
train, test = do_fe.doit(train, test)

whole_col = column_selector.get_whole_col()
test_col = column_selector.get_test_col()
train = train[whole_col]
test = test[test_col]
# print(train.head())
# print(train.describe())
# print(test.head())
# print(test.describe())

train.to_csv(PRED_TRAIN)
test.to_csv(PRED_TEST)
