import pandas as pd


def doit(train, test, train_active, test_active, train_period, test_period):

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

        a_df = get_meta_text(a_df)

    return train, test


def get_max_target(train, test):
    max_map = train.groupby("parent_category_name")["deal_probability"].agg("max").reset_index()
    max_map.columns = ["parent_category_name", "parent_max_deal_prob"]

    train = pd.merge(train, max_map, how="left", on="parent_category_name")
    test = pd.merge(test, max_map, how="left", on="parent_category_name")
    return train, test


def get_user_feature(df, all_df, all_period_df):
    all_period_df['days_up'] = all_period_df['date_to'].dt.dayofyear - all_period_df['date_from'].dt.dayofyear
    period_grouped = all_period_df.groupby("item_id")["days_up"].agg(["sum", "mean", "count"]).reset_index()
    period_grouped.columns = ["item_id", "items_dayup_sum", "items_dayup_mean", "items_dayup_count"]

    all_df = pd.merge(all_df, period_grouped, on="item_id", how="left")
    all_df["user_item_dayup_sum"] = all_df.groupby("user_id")["items_dayup_sum"].transform("mean")
    all_df["user_item_dayup_mean"] = all_df.groupby("user_id")["items_dayup_mean"].transform("mean")
    all_df["user_item_dayup_count"] = all_df.groupby("user_id")["items_dayup_count"].transform("mean")

    all_df["user_item_count"] = all_df.groupby("user_id")["item_id"].transform("count")
    all_df["user_max_seq"] = all_df.groupby("user_id")["item_seq_number"].transform("max")




def get_meta_text(a_df):
    russian_caps = "[АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ]"
    for cols in ["title", "description"]:
        print(cols)
        a_df[cols] = a_df[cols].astype(str).fillna('missing')
        a_df[cols + "_upper_count"] = a_df[cols].str.count(russian_caps)
        a_df[cols] = a_df[cols].str.lower()
        a_df[cols + '_num_chars'] = a_df[cols].apply(len)
        a_df[cols + '_num_words'] = a_df[cols].apply(lambda comment: len(comment.split()))  # Count number of Words
        a_df[cols + '_num_unique_words'] = a_df[cols].apply(lambda comment: len(set(w for w in comment.split())))
        a_df[cols + '_words_vs_unique'] = a_df[cols + '_num_unique_words'] / a_df[cols + '_num_words'] * 100
        a_df[cols + '_upper_char_share'] = a_df[cols + "_upper_count"] / a_df[cols + "_num_chars"] * 100
        a_df[cols + '_upper_word_share'] = a_df[cols + "_upper_count"] / a_df[cols + "_num_words"] * 100
    return a_df


def get_price_feature(df, all_df, name, grouping):
    print(name)
    avg_price_col_name = name + "_" + "avg_price"
    std_price_col_name = name + "_" + "std_price"
    scaled_price_col_name = name + "_" + "std_scale_price"
    price_cnt_col_name = name + "_" + "price_cnt"
    rolling_4_col_name = name + "_" + "rolling_price"
    forward_4_col_name = name + "_" + "forward_price"

    mapping_df = all_df.groupby(grouping)["price"].agg(["mean", "std", "count"]).reset_index()
    map_cols = grouping + [avg_price_col_name, std_price_col_name, price_cnt_col_name]
    mapping_df.columns = map_cols

    df = pd.merge(df, mapping_df, on=grouping, how="left")
    df[scaled_price_col_name] = df["price"] - df[avg_price_col_name]
    df[scaled_price_col_name] = df[scaled_price_col_name] / df[std_price_col_name]

    df[rolling_4_col_name] = df.groupby(grouping)["price"].transform(
        lambda g: g.rolling(4, min_periods=1).mean())
    df[forward_4_col_name] = df[rolling_4_col_name].shift(3)

    return df
