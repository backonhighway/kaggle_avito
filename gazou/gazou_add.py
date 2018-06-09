import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "avito")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
IMAGE_DIR = os.path.join(INPUT_DIR, "image_train")
ORG_TRAIN = os.path.join(INPUT_DIR, "train.csv")
ORG_TEST = os.path.join(INPUT_DIR, "test.csv")
# TRAIN_JPG = os.path.join(INPUT_DIR, "train_jpg.zip")
# TEST_JPG = os.path.join(INPUT_DIR, "test_jpg.zip")
IMAGE_FE_TRAIN = os.path.join(OUTPUT_DIR, "image_train.csv")

import numpy as np
import pandas as pd
import scipy.sparse
import gc
from avito.common import pocket_timer, pocket_logger, column_selector
from avito.fe import big_bow
import dask.dataframe as dd
from dask.multiprocessing import get
import multiprocessing
from multiprocessing import Pool
from PIL import Image as Im
from PIL import ExifTags
from skimage import feature
import operator
from collections import defaultdict
from scipy.stats import itemfreq
import cv2


def parallel_process(df, func):
    df_list = np.array_split(df, 90)
    pool = Pool(30)
    df = pd.concat(pool.map(func, df_list))
    pool.close()
    pool.join()
    return df



def get_exef(image):
    filename = os.path.join(IMAGE_DIR, image)
    img = Im.open(filename)
    exif = img._getexif()
    if exif is None:
        return
    exif_table = {}
    for tag_id, value in exif.items():
        tag = ExifTags.TAGS.get(tag_id, tag_id)
        exif_table[tag] = value
        print(tag)
        print(value)


def do_it_all(df):
    # df = parallel_process(df, do_size)
    # timer.time("done size")
    df["image"].apply(get_exef)
    return df


logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)

bad_files = ['4f029e2a00e892aa2cac27d98b52ef8b13d91471f613c8d3c38e3f29d4da0b0c.jpg',
             '8513a91e55670c709069b5f85e12a59095b802877715903abef16b7a6f306e58.jpg',
             '60d310a42e87cdf799afcd89dc1b11ae3fdc3d0233747ec7ef78d82c87002e83.jpg',
             'b98b291bd04c3d92165ca515e00468fd9756af9a8f1df42505deed1dcfb5d7ae.jpg']
images = os.listdir(IMAGE_DIR)
features = pd.DataFrame()
features['image'] = images
features = features[~features["image"].isin(bad_files)]
features = features[:1000]
timer.time("start")

features = do_it_all(features)

# drop_col = ["temp_size", "average_color"]
# features.drop(drop_col, axis=1, inplace=True)
# features["image"] = features["image"].apply(lambda w: w.replace(".jpg", ""))
# print(features.head())
# timer.time("done")
# features.to_csv(IMAGE_FE_TRAIN, index=False)

