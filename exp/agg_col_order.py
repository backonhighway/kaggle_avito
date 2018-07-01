import pandas as pd

df = pd.DataFrame({
    "a": [3,4,3,3,4,4,5,5,4,3],
    "v": [10,10,3,3,3,10,4,10,10,10],
})

df["v1"] = df["v"]
df["v2"] = df["v1"] + 10
df["v3"] = df["v2"] + 10

use_col = ["v1", "v2", "v3"]
col_name = "_dayup_"
temp0 = df.groupby("a")[use_col].agg(["mean", "std", "count"]).reset_index()
print(temp0)
temp0.columns = ["a"] + [col_name + "sum_mean", col_name + "mean_mean", col_name + "count_mean"]

print(temp0)