import pandas as pd
from sklearn import preprocessing
import string
import numpy as np
from avito.fe import additional_fe
from avito.common import pocket_lda, column_selector


def doit(train, test, train_active, test_active, train_period, test_period, timer):

    train, test, train_active, test_active, all_train, all_test = do_prep(train, test, train_active, test_active)
    timer.time("done_prep")

    train, test = get_lda(train, test, all_train, all_test, timer)
    timer.time("done lda")

    train, test = additional_fe.get_prev_week_history(train, test)
    timer.time("done encoding features")

    train, test = do_seq_sorted(train, test, all_train, all_test)
    timer.time("done seq features")

    train, test = get_user_feature(train, test, train_active, test_active, train_period, test_period)
    timer.time("done user_features")

    train, test = get_max_target(train, test)
    timer.time("done max_target")

    train = get_meta_text(train)
    test = get_meta_text(test)
    timer.time("done meta_text")

    train, test = get_all_price_features(train, test, all_train, all_test, timer)
    timer.time("done price_features")

    return train, test


def get_lda(train, test, train_all, test_all, timer):
    lda_train, lda_test = prep_lda(train, test, timer, False)
    lda_train_all, lda_test_all = prep_lda(train_all, test_all, timer, True)

    train = pd.merge(train, lda_train, on="item_id", how="left")
    test = pd.merge(test, lda_test, on="item_id", how="left")
    train = pd.merge(train, lda_train_all, on="item_id", how="left")
    test = pd.merge(test, lda_test_all, on="item_id", how="left")

    return train, test


def prep_lda(df_train, df_test, timer, is_all):
    le = preprocessing.LabelEncoder()
    col = "user_id"
    le.fit(list(df_train[col].values.astype('str')) + list(df_test[col].values.astype('str')))
    df_train["user_num"] = le.transform(df_train[col].values.astype('str'))
    df_test["user_num"] = le.transform(df_test[col].values.astype('str'))

    cols = ['user_num', 'city', 'image_top_1', 'param_all']
    lda_cols = ["lda_u", "lda_c", "lda_i", "lda_p"]
    for col, lda_col in zip(cols, lda_cols):
        df_train[lda_col] = df_train[col].fillna(0).astype(int)
        df_test[lda_col] = df_test[col].fillna(0).astype(int)

    maker = pocket_lda.GoldenLDA(timer, is_all)
    lda_train, lda_test = maker.create_features(df_train, df_test)

    df_train = pd.concat([df_train, lda_train], axis=1)
    df_test = pd.concat([df_test, lda_test], axis=1)

    use_col = column_selector.get_lda_col(is_all)
    use_col.append("item_id")
    return df_train[use_col], df_test[use_col]


def do_prep_vanilla(df):
    df["activation_date"] = pd.to_datetime(df["activation_date"])
    df["activation_dow"] = df["activation_date"].dt.dayofweek
    df["activation_day"] = df["activation_date"].dt.day

    # df["no_price"] = np.where(df["price"].isnull(), 1, 0)
    df["image_top_1_num"] = df["image_top_1"]
    df["price_last_digit"] = df["price"] % 10
    df["image_cat"] = df["image_top_1"].fillna(-1)
    return df


def do_prep(train, test, train_active, test_active):
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

    all_train = pd.concat([train, train_active]).reset_index()
    all_test = pd.concat([test, test_active]).reset_index()
    return train, test, train_active, test_active, all_train, all_test


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


def do_seq_sorted(train, test, train_all, test_all):
    train = get_sorted_feature(train, train_all)
    test = get_sorted_feature(test, test_all)
    return train, test


def get_sorted_feature(t, all_t_df):
    all_t = all_t_df.sort_values("item_seq_number")
    all_t["seq_diff"] = all_t.groupby("user_id")["item_seq_number"].shift(1)
    all_t["seq_diff"] = all_t["item_seq_number"] - all_t["seq_diff"]

    all_t["prev_cat1"] = all_t.groupby(["user_id"])["parent_category_name"].shift(1)
    all_t["prev_cat2"] = all_t.groupby(["user_id"])["category_name"].shift(1)
    all_t["prev_cat3"] = all_t.groupby(["user_id"])["param_1"].shift(1)
    mask = (all_t["prev_cat1"] == all_t["parent_category_name"]) & \
           (all_t["prev_cat2"] == all_t["category_name"]) & \
           (all_t["prev_cat3"] == all_t["param_1"])
    all_t["prev_is_same_pcat"] = np.where(all_t["prev_cat1"] == all_t["parent_category_name"], 1, 0)
    all_t["prev_is_same_ccat"] = np.where(all_t["prev_cat2"] == all_t["category_name"], 1, 0)
    all_t["prev_is_same_cat"] = np.where(mask, 1, 0)

    group_col = ["user_id", "parent_category_name", "category_name", "param_1"]
    all_t["user_pcat_nunique"] = all_t.groupby(["user_id"])["parent_category_name"].transform("nunique")
    all_t["user_ccat_nunique"] = all_t.groupby(["user_id"])["category_name"].transform("nunique")
    all_t["user_param_nunique"] = all_t.groupby(["user_id"])["param_1"].transform("nunique")
    all_t["user_city_nunique"] = all_t.groupby(["user_id"])["city"].transform("nunique")
    all_t["user_count"] = all_t.groupby(["user_id"])["item_id"].transform("count")
    all_t["user_p1_count"] = all_t.groupby(group_col)["item_id"].transform("count")
    all_t["same_user_cat_ratio"] = all_t["user_p1_count"] / all_t["user_count"] * 100

    pcat = ["user_id", "parent_category_name"]
    all_t["user_pcat_count"] = all_t.groupby(pcat)["item_id"].transform("count")
    ccat = ["user_id", "category_name"]
    all_t["user_ccat_count"] = all_t.groupby(ccat)["item_id"].transform("count")
    p2cat = ["user_id", "param_2"]
    all_t["user_p2_count"] = all_t.groupby(p2cat)["item_id"].transform("count")
    p3cat = ["user_id", "param_3"]
    all_t["user_p3_count"] = all_t.groupby(p3cat)["item_id"].transform("count")

    all_t["prev_price"] = all_t.groupby(["user_id"])["price"].shift(1)
    all_t["price_diff"] = all_t["price"] - all_t["prev_price"]

    all_t["prev_price_cat"] = all_t.groupby(group_col)["price"].shift(1)
    all_t["price_diff_cat"] = all_t["price"] - all_t["prev_price_cat"]

    all_t["user_max_seq"] = all_t.groupby("user_id")["item_seq_number"].transform("max")
    all_t["user_min_seq"] = all_t.groupby("user_id")["item_seq_number"].transform("min")
    all_t["user_seq_gap"] = all_t["user_max_seq"] - all_t["user_min_seq"]

    use_col = ["item_id", "seq_diff", "user_max_seq", "user_min_seq", "user_seq_gap",
               "user_pcat_nunique", "user_ccat_nunique", "user_param_nunique", "user_city_nunique",
               "prev_is_same_cat", "same_user_cat_ratio",
               "price_diff", "prev_price", "price_diff_cat", "prev_price_cat",
               "user_pcat_count", "user_ccat_count", "user_p1_count", "user_p2_count", "user_p3_count"]
    all_t = all_t[use_col]

    ret_df = pd.merge(t, all_t, on="item_id", how="left")
    return ret_df


def get_user_feature(train, test, train_all, test_all, train_period, test_period):
    train = make_active_period_features(train, train_all, train_period)
    test = make_active_period_features(test, test_all, test_period)

    train["user_item_count"] = train.groupby("user_id")["item_id"].transform("count")
    train["user_img_count"] = train.groupby("user_id")["image"].transform("count")
    train["user_image_count"] = train.groupby(["user_id", "image_top_1"])["item_id"].transform("count")
    train["user_image_nunique"] = train.groupby(["user_id"])["image_top_1"].transform("nunique")
    group_col = ["user_id", "image_top_1", "param_1"]
    train["user_image_cat_count"] = train.groupby(group_col)["item_id"].transform("count")
    # train["user_max_seq"] = train.groupby("user_id")["item_seq_number"].transform("max")
    # train["user_min_seq"] = train.groupby("user_id")["item_seq_number"].transform("min")
    test["user_item_count"] = test.groupby("user_id")["item_id"].transform("count")
    test["user_img_count"] = test.groupby("user_id")["image"].transform("count")
    test["user_image_count"] = test.groupby(["user_id", "image_top_1"])["item_id"].transform("count")
    test["user_image_nunique"] = test.groupby(["user_id"])["image_top_1"].transform("nunique")
    test["user_image_cat_count"] = test.groupby(group_col)["item_id"].transform("count")
    # test["user_max_seq"] = test.groupby("user_id")["item_seq_number"].transform("max")
    # test["user_min_seq"] = test.groupby("user_id")["item_seq_number"].transform("min")

    return train, test


def make_active_period_features(df, df_all, df_period):
    df_period['days_up'] = df_period['date_to'].dt.dayofyear - df_period['date_from'].dt.dayofyear
    period_grouped = df_period.groupby("item_id")["days_up"].agg(["sum", "mean", "count", "min", "max"]).reset_index()
    print(period_grouped.columns)

    df_all = pd.merge(df_all, period_grouped, on="item_id", how="left")
    user_period_df = get_period_df(df_all, ["user_id"], "u")
    uc_period_df = get_period_df(df_all, ["user_id", "category_name"], "uc")
    ucp1_period_df = get_period_df(df_all, ["user_id", "category_name", "param_1"], "up1")
    user_counts = df_all.groupby("user_id")["item_id"].count().reset_index()
    user_counts.columns = ["user_id", "user_item_count_all"]
    user_seq = df_all.groupby("user_id")["item_seq_number"].agg({"max", "min"}).reset_index()
    user_seq.columns = ["user_id", "user_max_seq_all", "user_min_seq_all"]
    user_period_df = pd.merge(user_period_df, user_counts, on="user_id", how="left")
    user_period_df = pd.merge(user_period_df, user_seq, on="user_id", how="left")
    df = pd.merge(df, user_period_df, on="user_id", how="left")
    df = pd.merge(df, uc_period_df, on=["user_id", "category_name"], how="left")
    df = pd.merge(df, ucp1_period_df, on=["user_id", "category_name", "param_1"], how="left")

    return df


def get_period_df(all_df, group_col, group_name):
    use_col = ["sum", "mean", "count"]
    col_name = group_name + "_dayup_"
    temp0 = all_df.groupby(group_col)[use_col].agg("mean").reset_index()
    temp0.columns = group_col + [col_name + "sum_mean", col_name + "mean_mean", col_name + "count_mean"]
    temp1 = all_df.groupby(group_col)[use_col].agg("std").reset_index()
    temp1.columns = group_col + [col_name + "sum_std", col_name + "mean_std", col_name + "count_std"]
    temp2 = all_df.groupby(group_col)[use_col].agg("sum").reset_index()
    temp2.columns = group_col + [col_name + "sum_sum", col_name + "mean_sum", col_name + "count_sum"]
    temp3 = all_df.groupby(group_col)["max"].agg("max").reset_index()
    temp3.columns = group_col + [col_name + "max"]
    temp4 = all_df.groupby(group_col)["min"].agg("min").reset_index()
    temp4.columns = group_col + [col_name + "min"]

    ret_df = pd.merge(temp0, temp1, on=group_col, how="left")
    ret_df = pd.merge(ret_df, temp2, on=group_col, how="left")
    ret_df = pd.merge(ret_df, temp3, on=group_col, how="left")
    ret_df = pd.merge(ret_df, temp4, on=group_col, how="left")

    return ret_df


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
        #"pc123r": ["parent_category_name", "category_name", "param_all", "region"],
        #"pc123": ["parent_category_name", "category_name", "param_all"],
        "u": ["user_id"],
        #"upc123": ["user_id", "parent_category_name", "category_name", "param_all"] ,
    }
    for name, grouping in group_list.items():
        train = get_price_feature(train, all_train, name, grouping)
        test = get_price_feature(test, all_test, name, grouping)
        timer.time("done " + name)

    imaged_group_list = {
        #"pc123ic": ["parent_category_name", "category_name", "param_all", "image_cat", "city"],
        #"pc123ir": ["parent_category_name", "category_name", "param_all", "image_cat", "region"],
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

