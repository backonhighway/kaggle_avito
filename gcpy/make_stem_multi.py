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
from avito.common import pocket_timer, pocket_logger, column_selector
from avito.fe import big_bow
import dask.dataframe as dd
from dask.multiprocessing import get

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
#dtypes = csv_loader.get_featured_dtypes()
#predict_col = column_selector.get_predict_col()

train = pd.read_csv(ORG_TRAIN, nrows=1000*10)
test = pd.read_csv(ORG_TEST, nrows=1000*10)
train = dd.from_pandas(train, npartitions=16)
test = dd.from_pandas(test, npartitions=16)
import csv
# train = pd.read_csv(ORG_TRAIN, quoting=csv.QUOTE_NONE).compute()
# test = dd.read_csv(ORG_TEST)
timer.time("read csv")
print("-"*40)

from nltk.corpus import stopwords
from nltk import SnowballStemmer
from nltk.tokenize import word_tokenize
import string
stemmer = SnowballStemmer("russian")
tokenizer = word_tokenize("russian")
russian_stop = set(stopwords.words('russian') + list(string.punctuation))


def preprocess(text):
    stem_list = [stemmer.stem(word) for word in word_tokenize(text) if word.lower() not in russian_stop and not word.isdigit()]
    return " ".join(stem_list)



for col in ["title", "description"]:
    print(col)
    train[col] = str(train[col])
    test[col] = str(test[col])
    #train[col] = train[col].map_partitions(preprocess).compute(get=get)
    timer.time("done train")
    test[col] = test[col].apply(preprocess, meta=pd.Series(name='Column B'))
    timer.time("done test")
    # unique_words = set()
    # train[col].str.lower().split().apply(unique_words.update)
    # test[col].str.lower().split().apply(unique_words.update)
    # print(len(unique_words))
    # stemmed_words = [stemmer.stem(w) for w in unique_words]

train = train.compute()
timer.time("done train")
test = test.compute()
timer.time("done test")
print(train.head()["title"])
print(test.head()["title"])

train.to_csv(STEM_TRAIN, index=False)
test.to_csv(STEM_TEST, index=False)
