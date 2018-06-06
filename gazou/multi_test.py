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


def get_size(filename):
    filename = os.path.join(IMAGE_DIR, filename)
    st = os.stat(filename)
    return st.st_size


def get_dimensions(filename):
    filename = os.path.join(IMAGE_DIR, filename)
    img_size = Im.open(filename).size
    return img_size


def average_pixel_width(img):
    filename = os.path.join(IMAGE_DIR, img)
    im = Im.open(filename)
    im_array = np.asarray(im.convert(mode='L'))
    edges_sigma1 = feature.canny(im_array, sigma=3)
    apw = (float(np.sum(edges_sigma1)) / (im.size[0]*im.size[1]))
    return apw*100


def get_dominant_color(img):
    filename = os.path.join(IMAGE_DIR, img)
    img = cv2.imread(filename)
    arr = np.float32(img)
    pixels = arr.reshape((-1, 3))

    n_colors = 5
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, .1)
    flags = cv2.KMEANS_RANDOM_CENTERS
    _, labels, centroids = cv2.kmeans(pixels, n_colors, None, criteria, 10, flags)

    palette = np.uint8(centroids)
    quantized = palette[labels.flatten()]
    quantized = quantized.reshape(img.shape)

    dominant_color = palette[np.argmax(itemfreq(labels)[:, -1])]
    return dominant_color


def get_average_color(img):
    filename = os.path.join(IMAGE_DIR, img)
    img = cv2.imread(filename)
    average_color = [img[:, :, i].mean() for i in range(img.shape[-1])]
    return average_color


def get_blurrness_score(image):
    filename = os.path.join(IMAGE_DIR, image)
    image = cv2.imread(filename)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(image, cv2.CV_64F).var()
    return fm


def do_size(df):
    df['image_size'] = df['image'].apply(get_size)
    df['temp_size'] = df['image'].apply(get_dimensions)
    df['width'] = df['temp_size'].apply(lambda x: x[0])
    df['height'] = df['temp_size'].apply(lambda x: x[1])
    return df


def do_dull_blur(df):
    df['dullness'] = df['image'].apply(lambda x: perform_color_analysis(x, 'black'))
    df['blurrness'] = df['image'].apply(get_blurrness_score)
    return df


def do_dominant_color(df):
    df['dominant_color'] = df['image'].apply(get_dominant_color)
    df['dominant_red'] = df['dominant_color'].apply(lambda x: x[0]) / 255
    df['dominant_green'] = df['dominant_color'].apply(lambda x: x[1]) / 255
    df['dominant_blue'] = df['dominant_color'].apply(lambda x: x[2]) / 255
    return df


def do_average_color(df):
    df['average_color'] = df['image'].apply(get_average_color)
    df['average_red'] = df['average_color'].apply(lambda x: x[0]) / 255
    df['average_green'] = df['average_color'].apply(lambda x: x[1]) / 255
    df['average_blue'] = df['average_color'].apply(lambda x: x[2]) / 255
    return df


def do_it_all(df):
    df = parallel_process(df, do_size)
    timer.time("done size")
    df = parallel_process(df, do_dull_blur)
    timer.time("done dull_blur")
    df = parallel_process(df, do_average_color)
    timer.time("done average color")
    # df = parallel_process(df, do_dominant_color)
    # timer.time("done dominant color")
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
# features = features[:1000]
timer.time("start")

features = do_it_all(features)

drop_col = ["temp_size", "average_color"]
features.drop(drop_col, axis=1, inplace=True)
features["image"] = features["image"].apply(lambda w: w.replace(".jpg", ""))
print(features.head())
timer.time("done")
features.to_csv(IMAGE_FE_TEST, index=False)

