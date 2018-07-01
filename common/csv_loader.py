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

SPAIN_DIR = os.path.join(APP_ROOT, "spain")
SPAIN_TRAIN_ALL = os.path.join(SPAIN_DIR, "all_preds_train.csv.gz")
SPAIN_TEST_ALL = os.path.join(SPAIN_DIR, "all_preds_test.csv.gz")
SPAIN_TRAIN_WEAK = os.path.join(SPAIN_DIR, "spain_train_weak.csv")
SPAIN_TEST_WEAK = os.path.join(SPAIN_DIR, "spain_test_weak.csv")
SPAIN_TRAIN_STRONG = os.path.join(SPAIN_DIR, "spain_train_strong.csv")
SPAIN_TEST_STRONG = os.path.join(SPAIN_DIR, "spain_test_strong.csv")
# SPAIN_TRAIN1 = os.path.join(SPAIN_DIR, "1.1.preds_train_lgb1_R.csv.gz")
# SPAIN_TRAIN2_1 = os.path.join(SPAIN_DIR, "1.2.preds_train_lgb1_text_word.csv.gz")
SPAIN_TRAIN2_2 = os.path.join(SPAIN_DIR, "predictions", "1.2.preds_train_ridge_text_word.csv.gz")
SPAIN_TEST2_2 = os.path.join(SPAIN_DIR, "predictions", "1.2.preds_test_ridge_text_word.csv.gz")
# SPAIN_TRAIN3 = os.path.join(SPAIN_DIR, "1.3.preds_train_lgb1_im.csv.gz")
# SPAIN_TRAIN4 = os.path.join(SPAIN_DIR, "1.4.preds_train_catboost1.csv.gz")
# SPAIN_TRAIN5 = os.path.join(SPAIN_DIR, "1.5.preds_train_Fast1.csv.gz")
# SPAIN_TRAIN6 = os.path.join(SPAIN_DIR, "1.6.preds_train_RNN_v1.csv.gz")
# SPAIN_TRAIN7 = os.path.join(SPAIN_DIR, "1.7.preds_train_xgb1_tfidf.csv.gz")
# SPAIN_TRAIN8 = os.path.join(SPAIN_DIR, "1.8.preds_train_lgb2.csv.gz")
# SPAIN_TRAIN9 = os.path.join(SPAIN_DIR, "1.9.preds_train_lgb2_price_v2.csv.gz")
# SPAIN_TEST1 = os.path.join(SPAIN_DIR, "1.1.preds_test_lgb1_R.csv.gz")
# SPAIN_TEST2_1 = os.path.join(SPAIN_DIR, "1.2.preds_test_lgb1_text_word.csv.gz")
# SPAIN_TEST3 = os.path.join(SPAIN_DIR, "1.3.preds_test_lgb1_im.csv.gz")
# SPAIN_TEST4 = os.path.join(SPAIN_DIR, "1.4.preds_test_catboost1.csv.gz")
# SPAIN_TEST5 = os.path.join(SPAIN_DIR, "1.5.preds_test_Fast1.csv.gz")
# SPAIN_TEST6 = os.path.join(SPAIN_DIR, "1.6.preds_test_RNN_v1.csv.gz")
# SPAIN_TEST7 = os.path.join(SPAIN_DIR, "1.7.preds_test_xgb1_tfidf.csv.gz")
# SPAIN_TEST8 = os.path.join(SPAIN_DIR, "1.8.preds_test_lgb2.csv.gz")
# SPAIN_TEST9 = os.path.join(SPAIN_DIR, "1.9.preds_test_lgb2_price_v2.csv.gz")

TEREKA_TRAIN6 = os.path.join(SPAIN_DIR, "rnn_v65_base_61_fasttext_valid.npy")
TEREKA_TRAIN7 = os.path.join(SPAIN_DIR, "rnn_v65_base_61_fasttext_seq100_valid.npy")
TEREKA_TRAIN8 = os.path.join(SPAIN_DIR, "capsule_v67_with_image_remove_pocket_feature_valid.npy")
TEREKA_TRAIN9 = os.path.join(SPAIN_DIR, "capsule_v68_with_image_remove_pocket_feature_valid.npy")
TEREKA_TEST6 = os.path.join(SPAIN_DIR, "rnn_v65_base_61_fasttext_test.npy")
TEREKA_TEST7 = os.path.join(SPAIN_DIR, "rnn_v65_base_61_fasttext_seq100_test.npy")
TEREKA_TEST8 = os.path.join(SPAIN_DIR, "capsule_v67_with_image_remove_pocket_feature_test.npy")
TEREKA_TEST9 = os.path.join(SPAIN_DIR, "capsule_v68_with_image_remove_pocket_feature_test.npy")
TEREKA_TRAIN_CLEAN = os.path.join(SPAIN_DIR, "tereka_train_clean.csv")
TEREKA_TEST_CLEAN = os.path.join(SPAIN_DIR, "tereka_test_clean.csv")

POCKET_STRONG_DIR = os.path.join(APP_ROOT, "pocket", "strong")
POCKET_STRONG1_TRAIN = os.path.join(POCKET_STRONG_DIR, "strong1-1_cv.csv")
POCKET_STRONG2_TRAIN = os.path.join(POCKET_STRONG_DIR, "strong1-2_cv.csv")
POCKET_STRONG3_TRAIN = os.path.join(POCKET_STRONG_DIR, "strong1-3_cv.csv")
POCKET_STRONG4_TRAIN = os.path.join(POCKET_STRONG_DIR, "strong1-4_cv.csv")
POCKET_STRONG1_TEST = os.path.join(POCKET_STRONG_DIR, "strong1-1.csv")
POCKET_STRONG2_TEST = os.path.join(POCKET_STRONG_DIR, "strong1-2.csv")
POCKET_STRONG3_TEST = os.path.join(POCKET_STRONG_DIR, "strong1-3.csv")
POCKET_STRONG4_TEST = os.path.join(POCKET_STRONG_DIR, "strong1-4.csv")
POCKET_STRONG_ALL_TRAIN = os.path.join(POCKET_STRONG_DIR, "strong1_all_train.csv")
POCKET_STRONG_ALL_TEST = os.path.join(POCKET_STRONG_DIR, "strong1_all_test.csv")

import pandas as pd
import numpy as np


def load_spain():
    spain_train_weak = pd.read_csv(SPAIN_TRAIN_WEAK)
    spain_test_weak = pd.read_csv(SPAIN_TEST_WEAK)
    return spain_train_weak, spain_test_weak


def save_spain():
    spain_train_all = pd.read_csv(SPAIN_TRAIN_ALL)
    use_col = ["item_id", "pred_R", "pred_text_word", "pred_im", "pred_cat1",
               "pred_FT1", "pred_RNN1", "xgb1_tfidf", "pred_lgb2", "pred_price"]
    spain_train_all = spain_train_all[use_col]

    spain2_2 = pd.read_csv(SPAIN_TRAIN2_2)
    use_col = ["item_id", "pred_text_word"]
    spain2_2 = spain2_2[use_col]
    spain2_2.columns = ["item_id", "pred_text_word_ridge"]
    spain_train_all = pd.merge(spain_train_all, spain2_2, on="item_id", how="left")
    spain_train_all.to_csv(SPAIN_TRAIN_WEAK, index=False)

    spain_test_all = pd.read_csv(SPAIN_TEST_ALL)
    use_col = ["item_id", "pred_R", "pred_text_word", "pred_im", "pred_cat1",
               "pred_FT1", "pred_RNN1", "xgb1_tfidf", "pred_lgb2", "pred_price"]
    spain_test_all = spain_test_all[use_col]

    spain2_2 = pd.read_csv(SPAIN_TEST2_2)
    use_col = ["item_id", "pred_text_word"]
    spain2_2 = spain2_2[use_col]
    spain2_2.columns = ["item_id", "pred_text_word_ridge"]
    spain_test_all = pd.merge(spain_test_all, spain2_2, on="item_id", how="left")
    spain_test_all.to_csv(SPAIN_TEST_WEAK, index=False)


def load_spain_strong():
    spain_train = pd.read_csv(SPAIN_TRAIN_STRONG)
    spain_test = pd.read_csv(SPAIN_TEST_STRONG)
    spain_train.drop("deal_probability", axis=1, inplace=True)
    return spain_train, spain_test


def save_spain_strong():
    spain_train_all = pd.read_csv(SPAIN_TRAIN_ALL)
    # use_col = ["item_id", "pred_R", "pred_text_word", "pred_im", "pred_cat1",
    #            "pred_FT1", "pred_RNN1", "xgb1_tfidf", "pred_lgb2", "pred_price"]
    # spain_train_all = spain_train_all[use_col]

    spain2_2 = pd.read_csv(SPAIN_TRAIN2_2)
    use_col = ["item_id", "pred_text_word"]
    spain2_2 = spain2_2[use_col]
    spain2_2.columns = ["item_id", "pred_text_word_ridge"]
    spain_train_all = pd.merge(spain_train_all, spain2_2, on="item_id", how="left")
    spain_train_all.to_csv(SPAIN_TRAIN_STRONG, index=False)

    spain_test_all = pd.read_csv(SPAIN_TEST_ALL)
    # use_col = ["item_id", "pred_R", "pred_text_word", "pred_im", "pred_cat1",
    #            "pred_FT1", "pred_RNN1", "xgb1_tfidf", "pred_lgb2", "pred_price"]
    # spain_test_all = spain_test_all[use_col]

    spain2_2 = pd.read_csv(SPAIN_TEST2_2)
    use_col = ["item_id", "pred_text_word"]
    spain2_2 = spain2_2[use_col]
    spain2_2.columns = ["item_id", "pred_text_word_ridge"]
    spain_test_all = pd.merge(spain_test_all, spain2_2, on="item_id", how="left")
    spain_test_all.to_csv(SPAIN_TEST_STRONG, index=False)


def load_tereka_clean():
    train = pd.read_csv(TEREKA_TRAIN_CLEAN)
    test = pd.read_csv(TEREKA_TEST_CLEAN)
    return train, test


def save_tereka_clean():
    org_train = pd.read_csv(ORG_TRAIN)
    org_train = np.array(org_train["item_id"])
    tereka_train1 = np.load(TEREKA_TRAIN6)
    tereka_train2 = np.load(TEREKA_TRAIN7)
    tereka_train3 = np.load(TEREKA_TRAIN8)
    tereka_train4 = np.load(TEREKA_TRAIN9)
    train_stack = [org_train, tereka_train1, tereka_train2, tereka_train3, tereka_train4]

    org_test = pd.read_csv(ORG_TEST)
    org_test = np.array(org_test["item_id"])
    tereka_test1 = np.load(TEREKA_TEST6)
    tereka_test2 = np.load(TEREKA_TEST7)
    tereka_test3 = np.load(TEREKA_TEST8)
    tereka_test4 = np.load(TEREKA_TEST9)
    test_stack = [org_test, tereka_test1, tereka_test2, tereka_test3, tereka_test4]
    cols = ["item_id", "tereka1", "tereka2", "tereka3", "tereka4"]
    ret_train = pd.DataFrame.from_items(zip(cols, train_stack))
    ret_test = pd.DataFrame.from_items(zip(cols, test_stack))

    ret_train.to_csv(TEREKA_TRAIN_CLEAN, index=False)
    ret_test.to_csv(TEREKA_TEST_CLEAN, index=False)


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


def load_pocket_strong():
    train = pd.read_csv(POCKET_STRONG_ALL_TRAIN)
    test = pd.read_csv(POCKET_STRONG_ALL_TEST)
    return train, test


def save_pocket_strong():
    pocket_train1 = pd.read_csv(POCKET_STRONG1_TRAIN)
    pocket_train2 = pd.read_csv(POCKET_STRONG2_TRAIN)
    pocket_train3 = pd.read_csv(POCKET_STRONG3_TRAIN)
    pocket_train4 = pd.read_csv(POCKET_STRONG4_TRAIN)
    pocket_test1 = pd.read_csv(POCKET_STRONG1_TEST)
    pocket_test2 = pd.read_csv(POCKET_STRONG2_TEST)
    pocket_test3 = pd.read_csv(POCKET_STRONG3_TEST)
    pocket_test4 = pd.read_csv(POCKET_STRONG4_TEST)

    cols = ["pocket1", "pocket2", "pocket3", "pocket4"]
    train_list = [pocket_train1, pocket_train2, pocket_train3, pocket_train4]
    test_list = [pocket_test1, pocket_test2, pocket_test3, pocket_test4]
    for col, a_train, a_test in zip(cols, train_list, test_list):
        replace_col = ["item_id", col]
        a_train.columns = replace_col
        a_test.columns = replace_col
    ret_train = pd.concat(train_list, axis=1)
    ret_test = pd.concat(test_list, axis=1)
    print(ret_train.head())
    ret_train.to_csv(POCKET_STRONG_ALL_TRAIN)
    ret_test.to_csv(POCKET_STRONG_ALL_TEST)
    exit(0)



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