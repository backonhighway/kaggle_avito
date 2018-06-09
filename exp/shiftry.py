import pandas as pd
import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from sklearn.pipeline import FeatureUnion

df = pd.DataFrame({
    "a": [3,4,3,3,4,4,5,5,4,3],
    "v": [10,10,3,3,3,10,4,10,10,10],
})


df["next"] = df.groupby("a")["v"].shift(1)


print(df)