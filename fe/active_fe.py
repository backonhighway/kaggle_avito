import pandas as pd
from sklearn import preprocessing
import string


def doit(train, test, train_active, test_active, train_period, test_period, timer):

    train, test, train_active, test_active, all_df, all_periods, all_train, all_test = \
        do_prep(train, test, train_active, test_active, train_period, test_period)
    timer.time("done_prep")

    train, test = get_user_feature(train, test, all_df, all_periods)
    timer.time("done user_features")

    train, test = get_max_target(train, test)
    timer.time("done max_target")

    train = get_meta_text(train)
    test = get_meta_text(test)
    timer.time("done meta_text")

    train, test = get_all_price_features(train, test, all_train, all_test, timer)
    timer.time("done price_features")

    return train, test


def do_prep_vanilla(df):
    df["activation_date"] = pd.to_datetime(df["activation_date"])
    df["activation_dow"] = df["activation_date"].dt.dayofweek
    df["activation_day"] = df["activation_date"].dt.day

    # df["no_price"] = np.where(df["price"].isnull(), 1, 0)
    df["image_top_1_num"] = df["image_top_1"]
    df["price_last_digit"] = df["price"] % 10
    df["image_cat"] = df["image_top_1"].fillna(-1)
    return df


def do_prep(train, test, train_active, test_active, train_period, test_period):
    train = do_prep_vanilla(train)
    test = do_prep_vanilla(test)
    train = get_param_all(train)
    test = get_param_all(test)
    train_active = get_param_all(train_active)
    test_active = get_param_all(test_active)

    # "item_id", "user_id", title, description
    cat_cols = ["region", "city", "parent_category_name", "category_name",
                "param_1", "param_2", "param_3", "user_type", "param_all"]
    for col in cat_cols:
        print(col)
        le = preprocessing.LabelEncoder()
        # le.fit(list(train[col].values.astype('str')) + list(test[col].values.astype('str')))
        le.fit(
            list(train[col].values.astype('str')) + list(test[col].values.astype('str')) +
            list(train_active[col].values.astype('str')) + list(test_active[col].values.astype('str'))
        )
        train[col] = le.transform(train[col].values.astype('str'))
        test[col] = le.transform(test[col].values.astype('str'))
        train_active[col] = le.transform(train_active[col].values.astype('str'))
        test_active[col] = le.transform(test_active[col].values.astype('str'))

    all_df = pd.concat([train, test, train_active, test_active])
    all_periods = pd.concat([train_period, test_period])
    all_train = pd.concat([train, train_active])
    all_test = pd.concat([test, test_active])
    return train, test, train_active, test_active, all_df, all_periods, all_train, all_test


def get_param_all(df):
    param_list = ["param_1", "param_2", "param_3"]
    for some_param in param_list:
        df[some_param].fillna("_", inplace=True)
    df['param_all'] = df["param_1"] + df["param_2"] + df["param_3"]
    return df


def get_max_target(train, test):
    max_map = train.groupby("parent_category_name")["deal_probability"].agg("max").reset_index()
    max_map.columns = ["parent_category_name", "parent_max_deal_prob"]

    train = pd.merge(train, max_map, how="left", on="parent_category_name")
    test = pd.merge(test, max_map, how="left", on="parent_category_name")

    return train, test


def get_user_feature(train, test, all_df, all_period_df):
    all_period_df['days_up'] = all_period_df['date_to'].dt.dayofyear - all_period_df['date_from'].dt.dayofyear
    period_grouped = all_period_df.groupby("item_id")["days_up"].agg(["sum", "mean", "count"]).reset_index()
    period_grouped.columns = ["item_id", "items_dayup_sum", "items_dayup_mean", "items_dayup_count"] #TODO min,max

    all_df = pd.merge(all_df, period_grouped, on="item_id", how="left")
    user_grouped = all_df.groupby("user_id")["items_dayup_sum", "items_dayup_mean", "items_dayup_count"].\
        agg("mean").reset_index()
    user_grouped.columns = ["user_id", "user_item_dayup_sum", "user_item_dayup_mean", "user_item_dayup_count"]
    # all_df["user_item_dayup_sum"] = all_df.groupby("user_id")["items_dayup_sum"].transform("mean")
    # all_df["user_item_dayup_mean"] = all_df.groupby("user_id")["items_dayup_mean"].transform("mean")
    # all_df["user_item_dayup_count"] = all_df.groupby("user_id")["items_dayup_count"].transform("mean")

    user_counts = all_df.groupby("user_id")["item_id"].count().reset_index()
    user_counts.columns = ["user_id", "user_item_count_all"]
    user_max_seq = all_df.groupby("user_id")["item_seq_number"].max().reset_index()
    user_max_seq.columns = ["user_id", "user_max_seq_all"]
    user_grouped = pd.merge(user_grouped, user_counts, on="user_id", how="left")
    user_grouped = pd.merge(user_grouped, user_max_seq, on="user_id", how="left")
    train = pd.merge(train, user_grouped, on="user_id", how="left")
    test = pd.merge(test, user_grouped, on="user_id", how="left")

    train["user_item_count"] = train.groupby("user_id")["item_id"].transform("count")
    train["user_max_seq"] = train.groupby("user_id")["item_seq_number"].transform("max")
    test["user_item_count"] = test.groupby("user_id")["item_id"].transform("count")
    test["user_max_seq"] = test.groupby("user_id")["item_seq_number"].transform("max")

    return train, test


def get_meta_text(df):
    punctuation = set(string.punctuation)
    emoji = set()
    for cols in ["title", "description"]:
        df[cols] = df[cols].astype(str).fillna('')
        for s in df[cols]:
            for char in s:
                if char.isdigit() or char.isalpha() or char.isalnum() or char.isspace() or char in punctuation:
                    continue
                emoji.add(char)
    # print(''.join(emoji))

    # russian_caps = "[АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ]"
    for cols in ["title", "description"]:
        print(cols)
        df[cols + '_num_chars'] = df[cols].apply(len)
        df[cols + '_num_words'] = df[cols].apply(lambda x: len(x.split()))
        df[cols + '_num_digits'] = df[cols].apply(lambda x: sum(w.isdigit() for w in x))
        df[cols + '_num_caps'] = df[cols].apply(lambda x: sum(w.isupper() for w in x))
        df[cols + '_num_spaces'] = df[cols].apply(lambda x: sum(w.isspace() for w in x))
        df[cols + '_num_punctuations'] = df[cols].apply(lambda x: sum(w in punctuation for w in x))
        df[cols + '_num_emojis'] = df[cols].apply(lambda x: sum(w in emoji for w in x))

        df[cols] = df[cols].str.lower()
        df[cols + '_num_unique_words'] = df[cols].apply(lambda x: len(set(w for w in x.split())))

        num_col_name = ["digits", "caps", "spaces", "punctuations", "emojis"]
        ratio_div_col = ["chars", "words"]
        for num_cols in num_col_name:
            for rdc in ratio_div_col:
                ratio_col_name = cols + "_" + num_cols + "_div_" + rdc
                org_n_col = cols + "_num_" + num_cols
                org_divide_col = cols + "_num_" + rdc
                df[ratio_col_name] = df[org_n_col] / (df[org_divide_col] + 1) * 100

        df[cols + '_unique_div_words'] = df[cols + '_num_unique_words'] / (df[cols + '_num_words'] + 1) * 100

    return df


def get_all_price_features(train, test, all_train, all_test, timer):
    group_list = {
        "pc123c": ["parent_category_name", "category_name", "param_all", "city"],
        "pc123r": ["parent_category_name", "category_name", "param_all", "region"],
        "pc123": ["parent_category_name", "category_name", "param_all"],
    }
    for name, grouping in group_list.items():
        train = get_price_feature(train, all_train, name, grouping)
        test = get_price_feature(test, all_test, name, grouping)
        timer.time("done " + name)

    imaged_group_list = {
        "pc123ic": ["parent_category_name", "category_name", "param_all", "image_cat", "city"],
        "pc123ir": ["parent_category_name", "category_name", "param_all", "image_cat", "region"],
        "pc123i": ["parent_category_name", "category_name", "param_all", "image_cat"],
    }
    for name, grouping in imaged_group_list.items():
        train = get_price_feature(train, train, name, grouping)
        test = get_price_feature(test, test, name, grouping)
        timer.time("done " + name)

    return train, test


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

