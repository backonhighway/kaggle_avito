import os, sys
import pandas as pd
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "avito")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
TF_COLS = os.path.join(OUTPUT_DIR, "tf_col.csv")


def get_predict_col():
    # item_id, user_id, title, description, image, deal_probability
    pred_col = [
        "region", "city", "parent_category_name", "category_name",
        "param_1", "param_2", "param_3",
        "price", "item_seq_number",
        "activation_dow", #"activation_day",
        "user_type", "image_top_1",
        "param_all",
        "user_item_count", "user_max_seq",
        "user_item_count_all", "user_max_seq_all",
        "user_item_dayup_sum", "user_item_dayup_mean", "user_item_dayup_count",
        "parent_max_deal_prob",
        "image_top_1_num", "price_last_digit",
        #"pc123c_avg_price", "pc123c_std_price", "pc123c_std_scale_price",
        #"pc123r_avg_price", "pc123r_std_price", "pc123r_std_scale_price",
        #"pc123_avg_price", "pc123_std_price", "pc123_std_scale_price",
        #"pc123ic_avg_price", "pc123ic_std_price", "pc123ic_std_scale_price",
        #"pc123ir_avg_price", "pc123ir_std_price", "pc123ir_std_scale_price",
        #"pc123i_avg_price", "pc123i_std_price", "pc123i_std_scale_price",
    ]

    meta_word_list = []
    group_list = ["title", "description"]
    col_names = ["_upper_count", "_num_chars", "_num_words", "_num_unique_words",
                 "_words_vs_unique", "_upper_char_share", "_upper_word_share", ]
    for a_group_name in group_list:
        for col_name in col_names:
            the_col_name = a_group_name + col_name
            meta_word_list.append(the_col_name)
    pred_col.extend(meta_word_list)

    added_grouped_list = []
    group_list = ["pc123c", "pc123r", "pc123", "pc123ic", "pc123ir", "pc123i"]
    col_names = ["avg_price", "std_price", "std_scale_price", "price_cnt",
                 "rolling_price", "forward_price"]
    for a_group_name in group_list:
        for col_name in col_names:
            the_col_name = a_group_name + "_" + col_name
            added_grouped_list.append(the_col_name)
    #pred_col.extend(added_grouped_list)
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


def get_pred_tf_col():
    pred_col = get_predict_col()
    tf_col = pd.read_csv(TF_COLS, header=None)
    tf_col = list(tf_col[0])
    pred_col.extend(tf_col)
    return pred_col








