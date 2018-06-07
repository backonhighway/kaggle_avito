import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from sklearn.pipeline import FeatureUnion

df1 = pd.DataFrame({
    "a": [3,4,3,3,4,4,5,5,4,3],
    "v": [10,10,3,3,3,10,4,10,10,10],
})


def get_user_history(df):
    df["new"] = df.groupby("a")["v"].transform(
        lambda g: g.expanding().mean().shift(1))
    return df


df = get_user_history(df1)

print(df)