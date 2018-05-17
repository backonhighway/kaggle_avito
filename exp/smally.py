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
        "Winner of life",
    ]
})

X_train = dft["text"]
stop_word_lib = set(stopwords.words('english'))

def do_tfidf():
    russian_stop = set(stopwords.words('english'))
    tfidf_param = {
        "stop_words": russian_stop,
        "analyzer": 'word',
        "token_pattern": r'\w{1,}',
        "sublinear_tf": True,
        "dtype": np.float32,
        "norm": 'l2',
        # "min_df":5,
        # "max_df":.9,
        "smooth_idf": False
    }

    tf_vector = TfidfVectorizer(ngram_range=(1, 2), max_features=50, **tfidf_param)
    tf_vector.fit(list(X_train))
    texty = tf_vector.transform(X_train)

    print(texty)
    print(texty.toarray())
    print(tf_vector.get_feature_names())


def do_cnt():
    cnt_vector = CountVectorizer(min_df=0, stop_words=stop_word_lib)
    transed = cnt_vector.fit_transform(X_train)
    print(cnt_vector.get_feature_names())
    print(transed.toarray())


do_tfidf()
do_cnt()

unioned = FeatureUnion([
    ("colA", )
])

