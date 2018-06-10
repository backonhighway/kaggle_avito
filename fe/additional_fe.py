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


def get_prev_week_history(train, test):
    train["day_of_year"] = train["activation_date"].dt.dayofyear
    print(train["day_of_year"].describe())
    print(train.groupby(["day_of_year"])["item_id"].count())

    first_week = train[train["day_of_year"] <= 100]

    daily_df = train.groupby(["day_of_year", "user"])["deal_probability"].mean().reset_index()
    daily_df.columns = ["day_of_year", "user", "prev_week_dp"]
    daily_df["day_of_year"] = daily_df["day_of_year"] + 7
    print(daily_df.groupby(["day_of_year"])["item_id"].count())

    train = pd.merge(train, daily_df, on=["day_of_year", "user"], how="left")
    test["prev_week_dp"] = test["user_deal_prob"]

    return train, test

