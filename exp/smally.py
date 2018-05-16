import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords

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
X_train = dft["text"]
tf_vector.fit(list(X_train))
texty = tf_vector.transform(X_train)

print(texty)
print(texty.toarray())
print(tf_vector.get_feature_names())