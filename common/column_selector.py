def get_predict_col():
    # item_id, user_id, title, description, image, deal_probability
    pred_col = [
        "region", "city", "parent_category_name", "category_name",
        "param_1", "param_2", "param_3",
        "price", "item_seq_number",
        "activation_dow", "activation_day",
        "user_type", "image_top_1",
    ]
    return pred_col

def get_whole_col():
    # item_id, user_id, title, description, image, deal_probability
    whole_col = [
        "deal_probability"
    ]
    whole_col.extend(get_predict_col())
    return whole_col

