import pandas as pd
import numpy as np
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
    ]
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

unioned = FeatureUnion([
    ("colA", TfidfVectorizer(ngram_range=(1, 2), max_features=50, **tfidf_param)),
    ("colB", CountVectorizer(min_df=0, stop_words=stop_word_lib))
])
print("here")
unioned.fit(dft)
res_df = unioned.transform(dft)
f_names = unioned.get_feature_names()
print(f_names)
print(res_df.toarray())
print("here")
