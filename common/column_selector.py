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
        #"param_1", "param_2", "param_3",
        "price", "item_seq_number",
        "activation_dow", #"activation_day",
        "user_type", "image_top_1",
        "param_all",
        "user_item_count", "user_max_seq",
        "pc123c_avg_price", "pc123c_std_price", "pc123c_std_scale_price",
        "pc123r_avg_price", "pc123r_std_price", "pc123r_std_scale_price",
        "pc123_avg_price", "pc123_std_price", "pc123_std_scale_price",
    ]
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








