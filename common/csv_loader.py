import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "avito")
INPUT_DIR = os.path.join(APP_ROOT, "input")
ORG_TRAIN = os.path.join(INPUT_DIR, "train.csv")
ORG_TEST = os.path.join(INPUT_DIR, "test.csv")

TEREKA_DIR = os.path.join(APP_ROOT, "tereka")
TEREKA_TRAIN_DIR = os.path.join(TEREKA_DIR, "train")
TEREKA_TEST_DIR = os.path.join(TEREKA_DIR, "test")
TEREKA_TRAIN1 = os.path.join(TEREKA_TRAIN_DIR, "rnn_v46_custom_w2b_seq250.npy")
TEREKA_TRAIN2 = os.path.join(TEREKA_TRAIN_DIR, "rnn_v48_custom_w2b_seq250_with_all_img_features.npy")
TEREKA_TRAIN3 = os.path.join(TEREKA_TRAIN_DIR, "rnn_v51_with_image.npy")
TEREKA_TRAIN4 = os.path.join(TEREKA_TRAIN_DIR, "rnn_v52_with_little_fix.npy")
TEREKA_TRAIN5 = os.path.join(TEREKA_TRAIN_DIR, "rnn_v60_glu_add_user_price_features.npy")
TEREKA_TEST1 = os.path.join(TEREKA_TEST_DIR, "rnn_v46_custom_w2b_seq250.npy")
TEREKA_TEST2 = os.path.join(TEREKA_TEST_DIR, "rnn_v48_custom_w2b_seq250_with_all_img_features.npy")
TEREKA_TEST3 = os.path.join(TEREKA_TEST_DIR, "rnn_v51_with_image.npy")
TEREKA_TEST4 = os.path.join(TEREKA_TEST_DIR, "rnn_v52_with_little_fix.npy")
TEREKA_TEST5 = os.path.join(TEREKA_TEST_DIR, "rnn_v60_glu_add_user_price_features.npy")

POCKET_DIR = os.path.join(APP_ROOT, "pocket")
POCKET_TRAIN_DIR = os.path.join(POCKET_DIR, "train")
POCKET_TEST_DIR = os.path.join(POCKET_DIR, "test")
POCKET_TRAIN1 = os.path.join(POCKET_TRAIN_DIR, "0616_lgbm_cv_train.csv")
POCKET_TEST1 = os.path.join(POCKET_TEST_DIR, "0616_lgbm_cv.csv")
POCKET_TRAIN2 = os.path.join(POCKET_TRAIN_DIR, "0618_lgbm_vanilla_train.csv")
POCKET_TEST2 = os.path.join(POCKET_TEST_DIR, "0618_lgbm_vanilla.csv")
POCKET_TRAIN3 = os.path.join(POCKET_TRAIN_DIR, "0618_auc_train.csv")
POCKET_TEST3 = os.path.join(POCKET_TEST_DIR, "0618_auc.csv")

import pandas as pd
import numpy as np


def load_tereka():
    org_train = pd.read_csv(ORG_TRAIN)
    org_train = np.array(org_train["item_id"])
    tereka_train1 = np.load(TEREKA_TRAIN1)
    tereka_train2 = np.load(TEREKA_TRAIN2)
    tereka_train3 = np.load(TEREKA_TRAIN3)
    tereka_train4 = np.load(TEREKA_TRAIN4)
    tereka_train5 = np.load(TEREKA_TRAIN5)
    train_stack = [org_train, tereka_train1, tereka_train2, tereka_train3, tereka_train4, tereka_train5]

    org_test = pd.read_csv(ORG_TEST)
    org_test = np.array(org_test["item_id"])
    tereka_test1 = np.load(TEREKA_TEST1)
    tereka_test2 = np.load(TEREKA_TEST2)
    tereka_test3 = np.load(TEREKA_TEST3)
    tereka_test4 = np.load(TEREKA_TEST4)
    tereka_test5 = np.load(TEREKA_TEST5)
    test_stack = [org_test, tereka_test1, tereka_test2, tereka_test3, tereka_test4, tereka_test5]

    cols = ["item_id", "tereka1", "tereka2", "tereka3", "tereka4", "tereka5"]
    ret_train = pd.DataFrame.from_items(zip(cols, train_stack))
    ret_test = pd.DataFrame.from_items(zip(cols, test_stack))

    return ret_train, ret_test


def load_pocket():
    pocket_train1 = pd.read_csv(POCKET_TRAIN1)
    pocket_train2 = pd.read_csv(POCKET_TRAIN2)
    pocket_train3 = pd.read_csv(POCKET_TRAIN3)
    pocket_test1 = pd.read_csv(POCKET_TEST1)
    pocket_test2 = pd.read_csv(POCKET_TEST2)
    pocket_test3 = pd.read_csv(POCKET_TEST3)
    cols = ["item_id", "pocket1"]
    pocket_train1.columns = cols
    pocket_test1.columns = cols
    cols = ["item_id", "pocket2"]
    pocket_train2.columns = cols
    pocket_test2.columns = cols
    cols = ["item_id", "pocket3"]
    pocket_train3.columns = cols
    pocket_test3.columns = cols

    ret_train = pd.merge(pocket_train1, pocket_train2, on="item_id", how="left")
    ret_train = pd.merge(ret_train, pocket_train3, on="item_id", how="left")
    ret_test = pd.merge(pocket_test1, pocket_test2, on="item_id", how="left")
    ret_test = pd.merge(ret_test, pocket_test3, on="item_id", how="left")

    return ret_train, ret_test


def get_dtypes():
    dtypes = {
        'ip': 'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8',
        'click_id': 'int32'
    }
    return dtypes


def get_featured_dtypes():

    dtypes = {
        'ip': 'uint32',
        'app': 'uint16',
        'device': 'uint16',
        'os': 'uint16',
        'channel': 'uint16',
        'is_attributed': 'uint8',
        'click_id': 'int32',
        'hour': 'uint16',
        'app_count': 'uint32',
        'os_count': 'uint32',
        'idoa_is_last_try': 'uint8',
        'ioa_is_last_try': 'uint8',
        'io_is_last_try': 'uint8',
        "group_i_hourly_count": "uint32",
        "group_i_hourly_count_share": "float32",
        'group_i_count': 'uint32',
        'group_ia_count': 'uint32',
        'group_io_count': 'uint32',
        'group_ic_count': 'uint32',
        'group_ido_count': 'uint32',
        'group_ioa_count': 'uint32',
        'group_idoa_count': 'uint32',
        'group_iac_count': 'uint32',
        'group_ioc_count': 'uint32',
        'group_ioac_count': 'uint32',
        'group_idoac_count': 'uint32',
        'group_i_nunique_os': 'uint32',
        'group_i_nunique_app': 'uint32',
        'group_i_nunique_channel': 'uint32',
        'group_i_nunique_device': 'uint32',
        'group_i_prev_click_time': 'float32',
        'group_i_next_click_time': 'float32',
        'group_ict_max': 'float32',
        'group_ict_std': 'float32',
        'group_ict_mean': 'float32',
        'group_ict_sum': 'float32',
        'group_io_nunique_app': 'uint32',
        'group_io_nunique_channel': 'uint32',
        'group_io_prev_click_time': 'float32',
        'group_io_next_click_time': 'float32',
        'group_ioct_max': 'float32',
        'group_ioct_std': 'float32',
        'group_ioct_mean': 'float32',
        'group_ioct_sum': 'float32',
        'group_ido_nunique_app': 'uint32',
        'group_ido_nunique_channel': 'uint32',
        'group_ido_prev_click_time': 'float32',
        'group_ido_next_click_time': 'float32',
        'group_idoct_max': 'float32',
        'group_idoct_std': 'float32',
        'group_idoct_mean': 'float32',
        'group_idoct_sum': 'float32',
        "group_i_top1_device_share": 'int16',
        "group_i_top2_device_share": 'int16',
        "group_ido_rolling_mean_prev_ct": 'float32'
    }
    return dtypes