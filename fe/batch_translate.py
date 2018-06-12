import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "avito")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
ORG_TRAIN = os.path.join(INPUT_DIR, "train.csv")
ORG_TEST = os.path.join(INPUT_DIR, "test.csv")
OUTPUT_TRAIN = os.path.join(OUTPUT_DIR, "translated_train.csv")
OUTPUT_TEST = os.path.join(OUTPUT_DIR, "translated_test.csv")
import pandas as pd
import sys
import textblob
from textblob import TextBlob
from textblob.translate import NotTranslated
from tqdm import tqdm,tqdm_pandas
from avito.common import csv_loader, column_selector, pocket_lgb, pocket_timer, pocket_logger


def translate(text):
    text = TextBlob(text)
    try:
        text = text.translate(from_lang="ru", to="en")
    except NotTranslated:
        pass

    return str(text)


def map_translate(x):
    timer.time("start translate")
    tqdm.pandas(tqdm())

    x["en_desc"] = x['description'].progress_map(translate)
    timer.time("done desc")

    x["en_title"] = x['title'].progress_map(translate)
    timer.time("done title")

    return x


logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
#train = pd.read_csv(ORG_TRAIN)
test = pd.read_csv(ORG_TEST)
#train["description"].fillna(" ", inplace=True)
#train["title"].fillna(" ", inplace=True)
test["description"].fillna(" ", inplace=True)
test["title"].fillna(" ", inplace=True)

#train = map_translate(train)
test = map_translate(test)

print("-------------")
#print(train["en_desc"])
#print(train["en_title"])
print(test["en_desc"])
print(test["en_title"])

use_col = ["en_desc", "en_title", "description", "title"]
#train = train[use_col]
#train.to_csv(OUTPUT_TRAIN, index=False)
test = test[use_col]
test.to_csv(OUTPUT_TEST, index=False)


# reader = pd.read_csv(INPUT_FILE, dtype=dtypes, usecols=use_cols, chunksize=1000*1000*5)
# print("done loading...")
#
# temp_df_list = []
# for tmp_df in reader:
#     print("next_chunk")
#     cst = pytz.timezone('Asia/Shanghai')
#     tmp_df['click_time'] = pd.to_datetime(tmp_df['click_time']).dt.tz_localize(pytz.utc).dt.tz_convert(cst)
#     tmp_df["day"] = tmp_df["click_time"].dt.day.astype('uint8')
#     tmp = tmp_df[tmp_df["day"] >= 9]
#     tmp.drop("day", axis=1, inplace=True)
#     if tmp is not None:
#         temp_df_list.append(tmp)

