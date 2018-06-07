import pandas as pd


def get_user_history(df):
    df["user_deal_prob"] = df.groupby("user_id")["deal_probability"].transform(
        lambda g: g.expanding().mean().shift(1))
    return df


def get_test_user_history(train, test):
    train_user_prob = train.groupby("user_id")["deal_probability"].mean().reset_index()
    train_user_prob.columns = ["user_id", "user_deal_prob"]
    test = pd.merge(test, train_user_prob, on="user_id", how="left")
    return test
