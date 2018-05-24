

def doit(train, test):
    # TODO all in one
    for a_df in (train, test):
        a_df["user_item_count"] = a_df.groupby("user_id")["item_id"].transform("count")
        a_df["user_max_seq"] = a_df.groupby("user_id")["item_seq_number"].transform("max")

        group_list = {
            "pc123c": ["parent_category_name", "category_name", "param_all", "city"],
            "pc123r": ["parent_category_name", "category_name", "param_all", "region"],
            "pc123": ["parent_category_name", "category_name", "param_all"],
            "pc123ic": ["parent_category_name", "category_name", "param_all", "image_top1", "city"],
            "pc123ir": ["parent_category_name", "category_name", "param_all", "image_top1", "region"],
            "pc123i": ["parent_category_name", "category_name", "param_all", "image_top1"],
        }
        for name, grouping in group_list.items():
            avg_price_col_name = name + "_" + "avg_price"
            std_price_col_name = name + "_" + "std_price"
            scaled_price_col_name = name + "_" + "std_scale_price"
            a_df[avg_price_col_name] = a_df.groupby(grouping)["price"].transform("mean")
            a_df[std_price_col_name] = a_df.groupby(grouping)["price"].transform("std")
            a_df[scaled_price_col_name] = a_df["price"] - a_df[avg_price_col_name]
            a_df[scaled_price_col_name] = a_df[scaled_price_col_name] / a_df[std_price_col_name]

    return train, test

