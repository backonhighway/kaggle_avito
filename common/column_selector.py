import os, sys
import pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "avito")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
from avito.common import filename_getter
DESC_TF_COLS, DESC_TF_TRAIN, DESC_TF_TEST = filename_getter.get_filename(OUTPUT_DIR, "desc", "tf")
TITLE_TF_COLS, TITLE_TF_TRAIN, TITLE_TF_TEST = filename_getter.get_filename(OUTPUT_DIR, "title", "tf")
TITLE_CNT_COLS, TITLE_CNT_TRAIN, TITLE_CNT_TEST = filename_getter.get_filename(OUTPUT_DIR, "title", "cnt")
DENSE_TF_COLS, DENSE_TF_TRAIN, DENSE_TF_TEST = filename_getter.get_filename(OUTPUT_DIR, "title_desc", "tf")
STEM_DESC_TF_COLS, STEM_DESC_TF_TRAIN, STEM_DESC_TF_TEST = filename_getter.get_filename(OUTPUT_DIR, "stem_desc", "tf")
STEM_TITLE_TF_COLS, STEM_TITLE_TF_TRAIN, STEM_TITLE_TF_TEST = filename_getter.get_filename(OUTPUT_DIR, "stem_title", "tf")
STEM_TITLE_CNT_COLS, STEM_TITLE_CNT_TRAIN, STEM_TITLE_CNT_TEST = filename_getter.get_filename(OUTPUT_DIR, "stem_title", "cnt")
STEM_DENSE_TF_COLS, STEM_DENSE_TF_TRAIN, STEM_DENSE_TF_TEST = filename_getter.get_filename(OUTPUT_DIR, "stem_title_desc", "tf")


def get_predict_col():
    # item_id, user_id, title, description, image, deal_probability
    pred_col = [
        "region", "city", "parent_category_name", "category_name",
        "param_1", "param_2", "param_3",
        "price", "item_seq_number",
        "activation_dow", #"activation_day",
        "user_type", "image_top_1",
        "param_all",
        "user_item_count", "user_img_count",
        "user_max_seq", "user_min_seq", "user_seq_gap", "user_max_seq_all", "user_min_seq_all",
        "user_item_count_all", "user_img_count",
        "user_item_dayup_sum", "user_item_dayup_mean", "user_item_dayup_count", "user_item_dayup_std",
        "parent_max_deal_prob",
        #"image_top_1_num", "price_last_digit",
        "image_size", "width", "height", "average_pixel_width",
        "average_red", "average_green", "average_blue",
        "dullness", "blurrness", "whiteness", #"image_timestamp_scaled",
        #"seq_diff",
        "prev_is_same_cat", "same_user_cat_count", "same_user_cat_ratio",
        "price_diff", "prev_price", "price_diff_cat", "prev_price_cat",
        "user_pcat_nunique", "user_ccat_nunique", "user_param_nunique", "user_city_nunique",
        "user_image_count", "user_image_nunique", "user_image_cat_count",
        "user_deal_prob",
        #"user_deal_prob_common", #"user_deal_prob",
        #"pc123c_avg_price", "pc123c_std_price", "pc123c_std_scale_price",
        #"pc123r_avg_price", "pc123r_std_price", "pc123r_std_scale_price",
        #"pc123_avg_price", "pc123_std_price", "pc123_std_scale_price",
        #"pc123ic_avg_price", "pc123ic_std_price", "pc123ic_std_scale_price",
        #"pc123ir_avg_price", "pc123ir_std_price", "pc123ir_std_scale_price",
        #"pc123i_avg_price", "pc123i_std_price", "pc123i_std_scale_price",
    ]

    meta_word_list = []
    group_list = ["title", "description"]
    for a_group_name in group_list:

        # base
        base_cols = ["_num_chars", "_num_words", "_num_digits", "_num_caps",
                     "_num_spaces", "_num_punctuations", "_num_emojis", "_num_unique_words"]
        for base_col in base_cols:
            the_col_name = a_group_name + base_col
            meta_word_list.append(the_col_name)

        # ratio
        num_col_name = ["digits", "caps", "spaces", "punctuations", "emojis"]
        ratio_div_col = ["chars", "words"]
        for num_cols in num_col_name:
            for rdc in ratio_div_col:
                ratio_col_name = a_group_name + "_" + num_cols + "_div_" + rdc
                #meta_word_list.append(ratio_col_name)
        # others
        other_cols = a_group_name + "_unique_div_words"
        meta_word_list.append(other_cols)

    pred_col.extend(meta_word_list)

    added_grouped_list = []
    group_list = ["pc123c", "pc123i", "u"]#, "upc123"]
    #group_list = ["pc123c", "pc123r", "pc123", "pc123ic", "pc123ir", "pc123i"]
    col_names = ["avg_price", "std_price", "std_scale_price", "price_cnt",
                 "rolling_price", "forward_price"]
    for a_group_name in group_list:
        for col_name in col_names:
            the_col_name = a_group_name + "_" + col_name
            added_grouped_list.append(the_col_name)
    pred_col.extend(added_grouped_list)
    return pred_col


def get_whole_col():
    # item_id, user_id, title, description, image, deal_probability
    whole_col = [
        "deal_probability",
    ]
    whole_col.extend(get_predict_col())
    return whole_col


def get_test_col():
    test_col = [
        "item_id"
    ]
    test_col.extend(get_predict_col())
    return test_col


def get_stem_col():
    pred_col = get_predict_col()
    desc_tf_col = pd.read_csv(STEM_DENSE_TF_COLS, header=None)
    desc_tf_col = list(desc_tf_col[0])
    desc_tf_col = ["desc" + str(c) for c in desc_tf_col]
    pred_col.extend(desc_tf_col)
    title_tf_col = pd.read_csv(STEM_TITLE_CNT_COLS, header=None)
    title_tf_col = list(title_tf_col[0])
    title_tf_col = ["title" + str(c) for c in title_tf_col]
    pred_col.extend(title_tf_col)
    return pred_col


def get_cols(file_a, file_b):
    pred_col = get_predict_col()
    desc_tf_col = pd.read_csv(file_a, header=None)
    desc_tf_col = list(desc_tf_col[0])
    desc_tf_col = ["desc" + str(c) for c in desc_tf_col]
    pred_col.extend(desc_tf_col)
    title_tf_col = pd.read_csv(file_b, header=None)
    title_tf_col = list(title_tf_col[0])
    title_tf_col = ["title" + str(c) for c in title_tf_col]
    pred_col.extend(title_tf_col)
    return pred_col


def get_pred_tf_col():
    pred_col = get_predict_col()
    desc_tf_col = pd.read_csv(DESC_TF_COLS, header=None)
    desc_tf_col = list(desc_tf_col[0])
    desc_tf_col = ["desc" + c for c in desc_tf_col]
    pred_col.extend(desc_tf_col)
    title_tf_col = pd.read_csv(TITLE_TF_COLS, header=None)
    title_tf_col = list(title_tf_col[0])
    title_tf_col = ["title" + str(c) for c in title_tf_col]
    pred_col.extend(title_tf_col)
    return pred_col


def get_pred_cnt_col():
    pred_col = get_predict_col()
    desc_tf_col = pd.read_csv(DESC_TF_COLS, header=None)
    desc_tf_col = list(desc_tf_col[0])
    desc_tf_col = ["desc" + c for c in desc_tf_col]
    pred_col.extend(desc_tf_col)
    title_tf_col = pd.read_csv(TITLE_CNT_COLS, header=None)
    title_tf_col = list(title_tf_col[0])
    title_tf_col = ["title" + str(c) for c in title_tf_col]
    pred_col.extend(title_tf_col)
    return pred_col


def get_pred_dense_col():
    pred_col = get_predict_col()
    desc_tf_col = pd.read_csv(DENSE_TF_COLS, header=None)
    desc_tf_col = list(desc_tf_col[0])
    desc_tf_col = ["dense" + c for c in desc_tf_col]
    pred_col.extend(desc_tf_col)
    return pred_col





