from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from nltk.corpus import stopwords
import numpy as np
import pandas as pd


def make_desc_tf(train, test):
    col = "description"
    train[col].fillna("missing", inplace=True)
    test[col].fillna("missing", inplace=True)
    train[col] = train[col].str.lower()
    test[col] = test[col].str.lower()
    all_desc_series = train[col].append(test[col])

    russian_stop = set(stopwords.words('russian'))
    tfidf_para = {
        "stop_words": russian_stop,
        "token_pattern": r'\w{1,}',
        # "sublinear_tf": True,
        "dtype": np.float32,
        # "smooth_idf": False
    }

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=17000, **tfidf_para)
    vectorizer.fit(all_desc_series)
    train_tf = vectorizer.transform(train[col])
    test_tf = vectorizer.transform(test[col])
    feature_names = vectorizer.get_feature_names()
    print(type(train_tf))

    # ret_train = pd.DataFrame(train_tf.toarray(), columns=feature_names)
    # ret_test = pd.DataFrame(test_tf.toarray(), columns=feature_names)
    # print(ret_df.head())
    return train_tf, test_tf, feature_names


def make_title_tf(train, test):
    col = "title"
    train[col].fillna("missing", inplace=True)
    test[col].fillna("missing", inplace=True)
    train[col] = train[col].str.lower()
    test[col] = test[col].str.lower()
    all_desc_series = train[col].append(test[col])

    russian_stop = set(stopwords.words('russian'))
    tfidf_para = {
        "stop_words": russian_stop,
        "token_pattern": r'\w{1,}',
        # "sublinear_tf": True,
        "dtype": np.float32,
        # "smooth_idf": False
    }

    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, **tfidf_para)
    vectorizer.fit(all_desc_series)
    train_tf = vectorizer.transform(train[col])
    test_tf = vectorizer.transform(test[col])
    feature_names = vectorizer.get_feature_names()
    print(type(train_tf))

    return train_tf, test_tf, feature_names
