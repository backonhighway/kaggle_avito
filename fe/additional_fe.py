import pandas as pd
import numpy as np


def get_user_history(df):
    df["user_deal_prob"] = df.groupby("user_id")["deal_probability"].transform(
        lambda g: g.expanding().mean().shift(1))
    return df


def get_test_user_history(train, test):
    train_user_prob = train.groupby("user_id")["deal_probability"].mean().reset_index()
    train_user_prob.columns = ["user_id", "user_deal_prob"]
    test = pd.merge(test, train_user_prob, on="user_id", how="left")
    return test


def user_is_in_both(train, test):
    train["is_common_user"] = np.where(train["user_id"].isin(test["user_id"]), 1, 0)
    test["is_common_user"] = np.where(test["user_id"].isin(train["user_id"]), 1, 0)
    return train, test


def get_history_if_in_both(train, test):
    train["user_deal_prob_common"] = np.where(train["is_common_user"] == 1, train["user_deal_prob"], np.NaN)
    test["user_deal_prob_common"] = test["user_deal_prob"]
    return train, test


def doit_all(train, test):
    train, test = user_is_in_both(train, test)
    train = get_user_history(train)
    test = get_test_user_history(train, test)
    train, test = get_history_if_in_both(train, test)

    return train, test


def get_dp(first_week, last_week, train, test, group_col, col_name):
    temp_df = first_week.groupby(group_col)["deal_probability"].mean().reset_index()
    temp_col = group_col + [col_name]
    temp_df.columns = temp_col
    last_week = pd.merge(last_week, temp_df, on=group_col, how="left")

    temp_df = train.groupby(group_col)["deal_probability"].mean().reset_index()
    temp_df.columns = temp_col
    test = pd.merge(test, temp_df, on=group_col, how="left")

    return last_week, test


def get_prev_week_history(train, test):
    train["day_of_year"] = train["activation_date"].dt.dayofyear
    print(train["day_of_year"].describe())
    print(train.groupby(["day_of_year"])["item_id"].count())

    first_week = train[train["day_of_year"] <= 80]
    last_week = train[train["day_of_year"] > 80]

    group_col = ["user_id"]
    last_week, test = get_dp(first_week, last_week, train, test, group_col, "prev_week_u_dp")

    group_col = ["user_id", "category_name"]
    last_week, test = get_dp(first_week, last_week, train, test, group_col, "prev_week_uc_dp")

    group_col = ["user_id", "category_name", "param_1"]
    last_week, test = get_dp(first_week, last_week, train, test, group_col, "prev_week_ucp1_dp")

    group_col = ["user_id", "image_top_1"]
    last_week, test = get_dp(first_week, last_week, train, test, group_col, "prev_week_ui_dp")

    group_col = ["image_top_1"]
    last_week, test = get_dp(first_week, last_week, train, test, group_col, "prev_week_i_dp")

    use_col = ["item_id", "prev_week_u_dp", "prev_week_uc_dp", "prev_week_ucp1_dp",
               "prev_week_ui_dp", "prev_week_i_dp"]
    last_week = last_week[use_col]
    train = pd.merge(train, last_week, on="item_id", how="left")

    return train, test

