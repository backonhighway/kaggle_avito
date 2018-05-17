from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion
from nltk.corpus import stopwords
import numpy as np

russian_stop = set(stopwords.words('russian'))

tfidf_para = {
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


def get_col(col_name): return lambda x: x[col_name]


vectorizer = FeatureUnion([
    ('description', TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=18000,
        **tfidf_para,
        preprocessor=get_col('description'))),
    ('text_feat', CountVectorizer(
        ngram_range=(1, 2),
        # max_features=7000,
        preprocessor=get_col('text_feat'))),
    ('title', TfidfVectorizer(
        ngram_range=(1, 2),
        **tfidf_para,
        # max_features=7000,
        preprocessor=get_col('title')))
])

vectorizer.fit(df.loc[traindex, :].to_dict('records'))
ready_df = vectorizer.transform(df.to_dict('records'))
tfvocab = vectorizer.get_feature_names()


# get char count
length_of_words = len(df["len"])