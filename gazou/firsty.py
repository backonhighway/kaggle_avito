import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "avito")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
ORG_TRAIN = os.path.join(INPUT_DIR, "train.csv")
ORG_TEST = os.path.join(INPUT_DIR, "test.csv")
TRAIN_JPG = os.path.join(INPUT_DIR, "train_jpg.zip")
TEST_JPG = os.path.join(INPUT_DIR, "test_jpg.zip")

import numpy as np
import pandas as pd
import scipy.sparse
import gc
from avito.common import pocket_timer, pocket_logger, column_selector
from avito.fe import big_bow
import dask.dataframe as dd
import multiprocessing
from multiprocessing import Pool


# def parallelize_dataframe(df, func):
#     a,b = np.array_split(df, num_partitions)
#     pool = Pool(num_cores)
#     df = pd.concat(pool.map(func, [a,b]))
#     pool.close()
#     pool.join()
#     return df


def average_pixel_width(img):
    path = images_path + img
    im = IMG.open(path)
    im_array = np.asarray(im.convert(mode='L'))
    edges_sigma1 = feature.canny(im_array, sigma=3)
    apw = (float(np.sum(edges_sigma1)) / (im.size[0]*im.size[1]))
    return apw*100

train = pd.read_csv(ORG_TRAIN, nrows=1000*1)
test = pd.read_csv(ORG_TEST, nrows=1000*1)

train = dd.from_pandas(train, npartitions=1)

train.apply()
