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
IMAGE_FE_TRAIN = os.path.join(OUTPUT_DIR, "image_train_compare.csv")

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



# def parallelize_dataframe(df, func):
#     df_list = np.array_split(df, num_partitions)
#     pool = Pool(num_cores)
#     df = pd.concat(pool.map(func, df_list))
#     pool.close()
#     pool.join()
#     return df

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





print("here")
logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)
# train = pd.read_csv(ORG_TRAIN, nrows=1000*1)
# test = pd.read_csv(ORG_TEST, nrows=1000*1)
#
# train = dd.from_pandas(train, npartitions=1)


images = os.listdir(IMAGE_DIR)
features = pd.DataFrame()
features['image'] = images
features = features[:100]
# features = dd.from_pandas(features, npartitions=10)
timer.time("start")

features['image_size'] = features['image'].apply(get_size)
features['temp_size'] = features['image'].apply(get_dimensions)
features['width'] = features['temp_size'].apply(lambda x : x[0])
features['height'] = features['temp_size'].apply(lambda x : x[1])
timer.time("done size")

features['dullness'] = features['image'].apply(lambda x : perform_color_analysis(x, 'black'))
timer.time("done color")

# features['dominant_color'] = features['image'].apply(get_dominant_color)
# features['dominant_red'] = features['dominant_color'].apply(lambda x: x[0]) / 255
# features['dominant_green'] = features['dominant_color'].apply(lambda x: x[1]) / 255
# features['dominant_blue'] = features['dominant_color'].apply(lambda x: x[2]) / 255
# timer.time("done dominant color")

features['average_color'] = features['image'].apply(get_average_color)
features['average_red'] = features['average_color'].apply(lambda x: x[0]) / 255
features['average_green'] = features['average_color'].apply(lambda x: x[1]) / 255
features['average_blue'] = features['average_color'].apply(lambda x: x[2]) / 255
timer.time("done average color")

features['blurrness'] = features['image'].apply(get_blurrness_score)
timer.time("done blurrness")

drop_col = ["temp_size", "average_color"]
features.drop(drop_col, axis=1, inplace=True)

print(features.head())
timer.time("done")
features.to_csv(IMAGE_FE_TRAIN)

