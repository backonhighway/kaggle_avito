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
from skimage import feature
import operator
from collections import defaultdict
from scipy.stats import itemfreq
import cv2


file_name="60d310a42e87cdf799afcd89dc1b11ae3fdc3d0233747ec7ef78d82c87002e83.jpg"
filename = os.path.join(IMAGE_DIR, file_name)
im = Im.open(filename)