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

dft = pd.DataFrame({
    "text": [
        "Gold medal",
        "Kaggle gold medal",
        "Kaggle grand master Danijel",
        "Master and doctor",
        "Winner of Kaggle",
    ],
    "num": [1, 2, 3, 4, 5]
})

X_train = dft["text"]
stop_word_lib = set(stopwords.words('english'))

tfidf_param = {
    "stop_words": stop_word_lib,
    "analyzer": 'word',
    "token_pattern": r'\w{1,}',
    "sublinear_tf": True,
    "dtype": np.float32,
    "norm": 'l2',
    # "min_df":5,
    # "max_df":.9,
    "smooth_idf": False
}

tfv = TfidfVectorizer(ngram_range=(1, 2), max_features=50, **tfidf_param)
ctv = CountVectorizer(min_df=0, stop_words=stop_word_lib)

print("here")
tfv.fit(X_train)
tvf_trans = tfv.transform(X_train)
ctv.fit(X_train)
ctv_trans = ctv.transform(X_train)

print(tvf_trans.shape)
csm = csr_matrix(dft[["num"]])
X = hstack([csm, tvf_trans, ctv_trans])
print(X)

