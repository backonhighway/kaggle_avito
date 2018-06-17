import numpy as np
import pandas as pd
import gc
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from typing import List, Tuple
from multiprocessing.pool import Pool
from functools import partial
import itertools


class GoldenLDA:
    def __init__(self, timer):
        self.timer = timer
        self.width = 5
        self.name = "LDA"

    def create_document_term_matrix(self, df, col1, col2):
        word_list = self.create_word_list(df, col1, col2)
        vectorizer = CountVectorizer()
        return vectorizer.fit_transform(word_list)

    def compute_latent_vectors(self, col_pair, train, test) -> Tuple[str, str, np.ndarray]:
        col1, col2 = col_pair
        all_df: pd.DataFrame = pd.concat([train, test])
        document_term_matrix = self.create_document_term_matrix(all_df, col1, col2)
        transformer = LatentDirichletAllocation(n_components=5, learning_method="online", random_state=99)
        self.timer.time("done transform")
        return col1, col2, transformer.fit_transform(document_term_matrix)

    def create_features(self, train, test) -> Tuple[pd.DataFrame, pd.DataFrame]:
        column_pairs = self.get_column_pairs()

        col1s = []
        col2s = []
        latent_vectors = []
        gc.collect()
        with Pool(15) as p:
            for col1, col2, latent_vector in p.map(
                    partial(self.compute_latent_vectors, train, test), column_pairs):
                col1s.append(col1)
                col2s.append(col2)
                latent_vectors.append(latent_vector.astype(np.float32))
        gc.collect()
        return self.get_feature(train, col1s, col2s, latent_vectors), \
               self.get_feature(test, col1s, col2s, latent_vectors)

    def get_feature(self, df_data: pd.DataFrame, cs1: List[str], cs2: List[str], vs: List[np.ndarray]) -> pd.DataFrame:
        features = np.zeros(shape=(len(df_data), len(cs1) * self.width), dtype=np.float32)
        columns = []
        for i, (col1, col2, latent_vector) in enumerate(zip(cs1, cs2, vs)):
            offset = i * self.width
            for j in range(self.width):
                columns.append(self.name + '-' + col1 + '-' + col2 + '-' + str(j))
            for j, val1 in enumerate(df_data[col1]):
                features[j, offset:offset + self.width] = latent_vector[val1]

        return pd.DataFrame(data=features, columns=columns)

    @staticmethod
    def get_column_pairs():
        columns = ['user_id', 'city', 'image_top_1', 'param_all']
        return [(col1, col2) for col1, col2 in itertools.product(columns, repeat=2) if col1 != col2]

    @staticmethod
    def create_word_list(df: pd.DataFrame, col1: str, col2: str) -> List[str]:
        col1_size = df[col1].max() + 1
        col2_list = [[] for _ in range(col1_size)]
        for val1, val2 in zip(df[col1], df[col2]):
            col2_list[val1].append(val2+10)  # 1-9 is a stop word

        return [' '.join(map(str, a_list)) for a_list in col2_list]