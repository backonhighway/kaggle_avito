import pandas as pd


def doit(train, test):

    max_map = train.groupby("parent_category_name")["deal_probability"].agg("max").reset_index()
    max_map.columns = ["parent_category_name", "parent_max_deal_prob"]
    # TODO all in one
    for a_df in train, test:
        a_df["user_item_count"] = a_df.groupby("user_id")["item_id"].transform("count")
        a_df["user_max_seq"] = a_df.groupby("user_id")["item_seq_number"].transform("max")
        a_df["image_cat"] = a_df["image_top_1"].fillna(-1)
        group_list = {
            "pc123c": ["parent_category_name", "category_name", "param_all", "city"],
            "pc123r": ["parent_category_name", "category_name", "param_all", "region"],
            "pc123": ["parent_category_name", "category_name", "param_all"],
            "pc123ic": ["parent_category_name", "category_name", "param_all", "image_cat", "city"],
            "pc123ir": ["parent_category_name", "category_name", "param_all", "image_cat", "region"],
            "pc123i": ["parent_category_name", "category_name", "param_all", "image_cat"],
        }
        for name, grouping in group_list.items():
            print(name)
            avg_price_col_name = name + "_" + "avg_price"
            std_price_col_name = name + "_" + "std_price"
            scaled_price_col_name = name + "_" + "std_scale_price"
            price_cnt_col_name = name + "_" + "price_cnt"
            rolling_4_col_name = name + "_" + "rolling_price"
            a_df[avg_price_col_name] = a_df.groupby(grouping)["price"].transform("mean")
            a_df[std_price_col_name] = a_df.groupby(grouping)["price"].transform("std")
            a_df[scaled_price_col_name] = a_df["price"] - a_df[avg_price_col_name]
            a_df[scaled_price_col_name] = a_df[scaled_price_col_name] / a_df[std_price_col_name]
            a_df[price_cnt_col_name] = a_df.groupby(grouping)["price"].transform("count")
            a_df[rolling_4_col_name] = a_df.groupby(grouping)["price"].transform(
                lambda g: g.rolling(4, min_periods=1).mean())

        a_df = get_meta_text(a_df)
        a_df = pd.merge(a_df, max_map, how="left", on="parent_category_name")

    train = pd.merge(train, max_map, how="left", on="parent_category_name")
    test = pd.merge(test, max_map, how="left", on="parent_category_name")
    return train, test


def get_meta_text(a_df):
    russian_caps = "[АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ]"
    for cols in ["title", "description"]:
        print(cols)
        a_df[cols] = a_df[cols].astype(str).fillna('missing')
        a_df[cols + "_upper_count"] = a_df[cols].str.count(russian_caps)
        a_df[cols] = a_df[cols].str.lower()
        a_df[cols + '_num_chars'] = a_df[cols].apply(len)
        a_df[cols + '_upper_share'] = a_df[cols + "_upper_count"] / a_df[cols + "_num_chars"] * 100
        a_df[cols + '_num_words'] = a_df[cols].apply(lambda comment: len(comment.split()))  # Count number of Words
        a_df[cols + '_num_unique_words'] = a_df[cols].apply(lambda comment: len(set(w for w in comment.split())))
        a_df[cols + '_words_vs_unique'] = a_df[cols + '_num_unique_words'] / a_df[cols + '_num_words'] * 100
    return a_df
