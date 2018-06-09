import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "avito")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
IMAGE_DIR = os.path.join(INPUT_DIR, "image_test")
ORG_TRAIN = os.path.join(INPUT_DIR, "train.csv")
ORG_TEST = os.path.join(INPUT_DIR, "test.csv")
# TRAIN_JPG = os.path.join(INPUT_DIR, "train_jpg.zip")
# TEST_JPG = os.path.join(INPUT_DIR, "test_jpg.zip")
IMAGE_FE_TRAIN = os.path.join(OUTPUT_DIR, "image_train.csv")
IMAGE_FE_TEST = os.path.join(OUTPUT_DIR, "image_test.csv")
IMAGE_FE_TRAIN_OUT = os.path.join(OUTPUT_DIR, "image_train_next.csv")
IMAGE_FE_TEST_OUT = os.path.join(OUTPUT_DIR, "image_test_next.csv")

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


def average_pixel_width(img):
    filename = os.path.join(IMAGE_DIR, img)
    im = Im.open(filename)
    im_array = np.asarray(im.convert(mode='L'))
    edges_sigma1 = feature.canny(im_array, sigma=3)
    apw = (float(np.sum(edges_sigma1)) / (im.size[0]*im.size[1]))
    return apw*100


def color_analysis(img):
    # obtain the color palette of the image
    palette = defaultdict(int)
    for pixel in img.getdata():
        palette[pixel] += 1

    # sort the colors present in the image
    sorted_x = sorted(palette.items(), key=operator.itemgetter(1), reverse = True)
    light_shade, dark_shade, shade_count, pixel_limit = 0, 0, 0, 25
    for i, x in enumerate(sorted_x[:pixel_limit]):
        if all(xx <= 20 for xx in x[0][:3]): ## dull : too much darkness
            dark_shade += x[1]
        if all(xx >= 240 for xx in x[0][:3]): ## bright : too much whiteness
            light_shade += x[1]
        shade_count += x[1]

    light_percent = round((float(light_shade ) /shade_count ) *100, 2)
    dark_percent = round((float(dark_shade ) /shade_count ) *100, 2)
    return light_percent, dark_percent


def perform_color_analysis(img, flag):
    filename = os.path.join(IMAGE_DIR, img)
    im = Im.open(filename)

    # cut the images into two halves as complete average may give bias results
    size = im.size
    halves = (size[0] / 2, size[1] / 2)
    im1 = im.crop((0, 0, size[0], halves[1]))
    im2 = im.crop((0, halves[1], size[0], size[1]))

    try:
        light_percent1, dark_percent1 = color_analysis(im1)
        light_percent2, dark_percent2 = color_analysis(im2)
    except Exception as e:
        return None

    light_percent = (light_percent1 + light_percent2) / 2
    dark_percent = (dark_percent1 + dark_percent2) / 2

    if flag == 'black':
        return dark_percent
    elif flag == 'white':
        return light_percent
    else:
        return None


def do_additional_1(df):
    df['whiteness'] = df['temp_dir'].apply(lambda x: perform_color_analysis(x, 'white'))
    return df


def do_additional_2(df):
    df['average_pixel_width'] = df['temp_dir'].apply(average_pixel_width)
    return df


def do_it_all(df):
    df["temp_dir"] = df["image"].apply(lambda w: w + ".jpg")
    df = parallel_process(df, do_additional_1)
    timer.time("done 1")
    df = parallel_process(df, do_additional_2)
    timer.time("done 2")
    df.drop(["temp_dir"], axis=1, inplace=True)
    return df


logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)

test = pd.read_csv(IMAGE_FE_TEST)
timer.time("start")

test = do_it_all(test)
print(test.head())

timer.time("done")
test.to_csv(IMAGE_FE_TEST_OUT, index=False)

