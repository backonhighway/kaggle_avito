import pandas as pd

df = pd.DataFrame({
    "a": [3,4,3,3,4,4,5,5,4,3],
    "v": [10,10,3,3,3,10,4,10,10,10],
})

three_index = df[df["a"] == 3].index.values
print(three_index)

temp = df.iloc[three_index]
print(temp)