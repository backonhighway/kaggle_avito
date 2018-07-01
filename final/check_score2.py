import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "avito")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
INPUT_DIR = os.path.join(APP_ROOT, "input")
SPAIN_DIR = os.path.join(APP_ROOT, "spain")
POCKET_DIR = os.path.join(APP_ROOT, "pocket", "strong")
ORG_TRAIN = os.path.join(INPUT_DIR, "train.csv")
PRED_TRAIN = os.path.join(OUTPUT_DIR, "pred_train.csv")
CV_FOLDS = os.path.join(SPAIN_DIR, "train_folds.csv")
SUB_DIR = os.path.join(APP_ROOT, "submission")
OUTPUT_PRED = os.path.join(SUB_DIR, "strong2.csv")
OUTPUT_CV_PRED = os.path.join(SUB_DIR, "strong2_cv.csv")
OUTPUT_PRED_TEMP = os.path.join(SUB_DIR, "strong2_temp.csv")
OUTPUT_CV_PRED_TEMP = os.path.join(SUB_DIR, "strong2_cv_temp.csv")
import pandas as pd
import numpy as np
from sklearn import metrics
from avito.common import pocket_timer, pocket_logger

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
# train = pd.read_csv(ORG_TRAIN)
# use_col = ["item_id", "deal_probability", "parent_category_name", "category_name",
#            "param_1", "param_2", "param_3"]
# train = train[use_col]
# df = pd.read_csv(OUTPUT_CV_PRED)
# df = pd.merge(df, train, on="item_id", how="left")
# df.to_csv(OUTPUT_CV_PRED_TEMP, index=False)
# exit(0)

df = pd.read_csv(OUTPUT_CV_PRED_TEMP)
timer.time("read csv")
from sklearn import model_selection
df, holdout = model_selection.train_test_split(df, test_size=0.1, random_state=99)

df["hope"] = df["cv_pred"]
p3_col = ["parent_category_name", "category_name", "param_1", "param_2", "param_3"]
p2_col = ["parent_category_name", "category_name", "param_1", "param_2"]
p1_col = ["parent_category_name", "category_name", "param_1"]
c_col = ["parent_category_name", "category_name"]

df["max_of_param3"] = df.groupby(p3_col)["deal_probability"].transform("max")
df["min_of_param3"] = df.groupby(p3_col)["deal_probability"].transform("min")
df["max_of_param2"] = df.groupby(p2_col)["deal_probability"].transform("max")
df["min_of_param2"] = df.groupby(p2_col)["deal_probability"].transform("min")
df["max_of_param1"] = df.groupby(p1_col)["deal_probability"].transform("max")
df["min_of_param1"] = df.groupby(p1_col)["deal_probability"].transform("min")
df["max_of_cat"] = df.groupby(c_col)["deal_probability"].transform("max")
df["min_of_cat"] = df.groupby(c_col)["deal_probability"].transform("min")

# df["hope"] = np.where(df["hope"] > df["max_of_param2"], df["max_of_param2"], df["hope"])
# df["hope"] = np.where(df["hope"] < df["min_of_param2"], df["min_of_param2"], df["hope"])
#df["hope"] = np.clip(df["hope"],0.0, df["max_of_param2"])
#print(df["hope"].describe())

df["maximum"] = np.where(
    df["param_3"].notnull(), df["max_of_param3"],
    np.where(
        df["param_2"].notnull(), df["max_of_param2"],
        np.where(
            df["param_1"].notnull(), df["max_of_param1"], df["max_of_cat"]
        )
    )
)

df["minimum"] = np.where(
    df["param_3"].notnull(), df["min_of_param3"],
    np.where(
        df["param_2"].notnull(), df["min_of_param2"],
        np.where(
            df["param_1"].notnull(), df["min_of_param1"], df["min_of_cat"]
        )
    )
)


df["base"] = np.where(df["hope"] > df["max_of_param3"], df["max_of_param3"], df["hope"])
#df["base"] = np.where(df["hope"] < df["min_of_param3"], df["min_of_param3"], df["hope"])
df["hope"] = np.where(df["hope"] > df["maximum"], df["maximum"], df["hope"])
#df["hope"] = np.where(df["hope"] < df["minimum"], df["minimum"], df["hope"])

print("---"*30)
y_pred = df["cv_pred"]
y_true = df["deal_probability"]
score = metrics.mean_squared_error(y_true, y_pred) ** 0.5
print(score)

y_pred = df["base"]
y_true = df["deal_probability"]
score = metrics.mean_squared_error(y_true, y_pred) ** 0.5
print(score)

y_pred = df["hope"]
y_true = df["deal_probability"]
score = metrics.mean_squared_error(y_true, y_pred) ** 0.5
print(score)

print("---"*30)
y_pred = df["cv_pred"]
y_true = df["deal_probability"]
score = metrics.mean_squared_error(y_true, y_pred) ** 0.5
print(score)

y_pred = df["base"]
y_true = df["deal_probability"]
score = metrics.mean_squared_error(y_true, y_pred) ** 0.5
print(score)

y_pred = df["hope"]
y_true = df["deal_probability"]
score = metrics.mean_squared_error(y_true, y_pred) ** 0.5
print(score)


