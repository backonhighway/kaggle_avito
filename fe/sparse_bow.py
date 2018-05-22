from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from nltk.corpus import stopwords
import numpy as np
import pandas as pd


def make_tfidf(train, test):
    train["description"].fillna("missing", inplace=True)
    test["description"].fillna("missing", inplace=True)
    all_series = train["description"].append(test["description"])

    russian_stop = set(stopwords.words('russian'))
    tfidf_para = {
        "stop_words": russian_stop,
        "token_pattern": r'\w{1,}',
        # "sublinear_tf": True,
        "dtype": np.float32,
        # "smooth_idf": False
    }

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=100, **tfidf_para)
    vectorizer.fit(all_series)
    # vectorizer.fit(train["description"])

    train_tf = vectorizer.transform(train["description"])
    test_tf = vectorizer.transform(test["description"])
    feature_names = vectorizer.get_feature_names()
    print(type(train_tf))

    # ret_train = pd.DataFrame(train_tf.toarray(), columns=feature_names)
    # ret_test = pd.DataFrame(test_tf.toarray(), columns=feature_names)
    # print(ret_df.head())
    return train_tf, test_tf, feature_names
