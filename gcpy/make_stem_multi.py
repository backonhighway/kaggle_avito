import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "avito")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
ORG_TRAIN = os.path.join(INPUT_DIR, "train.csv")
ORG_TEST = os.path.join(INPUT_DIR, "test.csv")
STEM_TRAIN = os.path.join(OUTPUT_DIR, "stem_train.csv")
STEM_TEST = os.path.join(OUTPUT_DIR, "stem_test.csv")
from avito.common import filename_getter
DESC_TF_COLS, DESC_TF_TRAIN, DESC_TF_TEST = filename_getter.get_filename(OUTPUT_DIR, "desc", "tf")
TITLE_TF_COLS, TITLE_TF_TRAIN, TITLE_TF_TEST = filename_getter.get_filename(OUTPUT_DIR, "title", "tf")
TITLE_CNT_COLS, TITLE_CNT_TRAIN, TITLE_CNT_TEST = filename_getter.get_filename(OUTPUT_DIR, "title", "cnt")
DENSE_TF_COLS, DENSE_TF_TRAIN, DENSE_TF_TEST = filename_getter.get_filename(OUTPUT_DIR, "title_desc", "tf")


import numpy as np
import pandas as pd
import scipy.sparse
import gc
from avito.common import pocket_timer, pocket_logger, row_parallel, column_selector
from avito.fe import big_bow
import dask.dataframe as dd
from dask.multiprocessing import get

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
#dtypes = csv_loader.get_featured_dtypes()
#predict_col = column_selector.get_predict_col()

train = pd.read_csv(ORG_TRAIN)
test = pd.read_csv(ORG_TEST)
# train = pd.read_csv(ORG_TRAIN, nrows=1000*10)
# test = pd.read_csv(ORG_TEST, nrows=1000*10)
timer.time("read csv")
print("-"*40)

from nltk.corpus import stopwords
from nltk import SnowballStemmer
from nltk.tokenize import word_tokenize
import string
stemmer = SnowballStemmer("russian")
tokenizer = word_tokenize("russian")
russian_stop = set(stopwords.words('russian') + list(string.punctuation))


def pre_process(text):
    stem_list = [stemmer.stem(word) for word in word_tokenize(text) if word.lower() not in russian_stop and not word.isdigit()]
    return " ".join(stem_list)


def do_it_all(df):
    for col in ["title", "description"]:
        new_col_name = "stem_" + col
        df[new_col_name] = df[col].apply(pre_process)
    return df


for str_col in ["title", "description"]:
    train[str_col] = train[str_col].astype(str)
    test[str_col] = test[str_col].astype(str)

processor = row_parallel.GoldenParallelProcessor(train, [do_it_all], ["train"], 30, timer)
processor.do_process()
train = processor.df

processor = row_parallel.GoldenParallelProcessor(test, [do_it_all], ["test"], 30, timer)
processor.do_process()
test = processor.df

use_col = ["item_id", "stem_title", "stem_description"]
train = train[use_col]
test = test[use_col]
print(train.head())
print(test.head())

train.to_csv(STEM_TRAIN, index=False)
test.to_csv(STEM_TEST, index=False)
