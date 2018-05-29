from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from nltk.corpus import stopwords
import numpy as np
import pandas as pd
import scipy.sparse


def save_sparsed(filenames, ret_dfs):
    temp_df = pd.DataFrame(ret_dfs[0])
    temp_df.to_csv(filenames[0], index=False, header=None)
    scipy.sparse.save_npz(filenames[1], ret_dfs[1])
    scipy.sparse.save_npz(filenames[2], ret_dfs[2])


def make_desc_tf(train, test):
    tfidf_param = get_tf_param()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=15000, **tfidf_param)

    return do_tf(train, test, "description", vectorizer)


def make_title_tf(train, test):
    tfidf_param = get_tf_param()
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=5000, **tfidf_param)

    return do_tf(train, test, "title", vectorizer)


def make_title_cnt(train, test):
    tfidf_param = get_tf_param()
    vectorizer = CountVectorizer(ngram_range=(1, 2), **tfidf_param)

    return do_tf(train, test, "title", vectorizer)


def get_tf_param():
    russian_stop = set(stopwords.words('russian'))
    tfidf_param = {
        "stop_words": russian_stop,
        "token_pattern": r'\w{1,}',
        # "sublinear_tf": True,
        "dtype": np.float32,
        # "smooth_idf": False
    }
    return tfidf_param


def do_tf(train, test, col_name, vectorizer):
    col = col_name
    train[col].fillna("missing", inplace=True)
    test[col].fillna("missing", inplace=True)
    train[col] = train[col].str.lower()
    test[col] = test[col].str.lower()
    all_desc_series = train[col].append(test[col])

    vectorizer.fit(all_desc_series)
    train_tf = vectorizer.transform(train[col])
    test_tf = vectorizer.transform(test[col])
    feature_names = vectorizer.get_feature_names()
    print(type(train_tf))
    # ret_train = pd.DataFrame(train_tf.toarray(), columns=feature_names)
    # ret_test = pd.DataFrame(test_tf.toarray(), columns=feature_names)
    # print(ret_df.head())

    return train_tf, test_tf, feature_names