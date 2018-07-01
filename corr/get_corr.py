import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "avito")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
INPUT_DIR = os.path.join(APP_ROOT, "input")
SPAIN_DIR = os.path.join(APP_ROOT, "spain")
blend_2147 = os.path.join(SPAIN_DIR, "2147.csv.gz")
blend_2148 = os.path.join(SPAIN_DIR, "2148.csv.gz")
robust = os.path.join(SPAIN_DIR, "subm_stack_level2_0_10.211352316876.csv.gz")

import pandas as pd
import numpy as np
from sklearn import metrics
from avito.common import pocket_timer, pocket_logger

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)

import pandas as pd
import numpy as np


blend_2147_df = pd.read_csv(blend_2147)
blend_2148_df = pd.read_csv(blend_2148)
robust_df = pd.read_csv(robust)
# pocket_df.columns=["pred"]
# mamas_df.columns=["pred"]
# danijel_df.columns=["pred"]
# print(pocket_df.describe())
# print(mamas_df.describe())
# print(danijel_df.describe())


def show_corr(df1, df2):
    corr_val = df1["deal_probability"].corr(df2["deal_probability"])
    return corr_val


def show_corr_spear(df1, df2, name):
    corr_val = df1["pred"].corr(df2["pred"], method="spearman")
    return name, corr_val

cor1 = show_corr(blend_2147_df, blend_2148_df)
cor2 = show_corr(blend_2147_df, robust_df)
print(cor1)
print(cor2)


