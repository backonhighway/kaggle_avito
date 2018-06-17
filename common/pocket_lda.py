import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from typing import List, Tuple


class LdaMaker:
    def __init__(self, timer):
        self.timer = timer

    def create_document_term_matrix(self, df, col1, col2):
        word_list = self.create_word_list(df, col1, col2)
        vectorizer = CountVectorizer()
        return vectorizer.fit_transform(word_list)

    def compute_latent_vectors(self, col_pair, train_path: str, test_path: str) -> Tuple[str, str, np.ndarray]:
        col1, col2 = col_pair
        df_train = pd.read_feather(train_path)
        df_test = pd.read_feather(test_path)
        df_data: pd.DataFrame = pd.concat([df_train, df_test])
        del df_train, df_test
        document_term_matrix = self.create_document_term_matrix(df_data, col1, col2)
        transformer = LatentDirichletAllocation(n_components=5, learning_method="online", random_state=99)

        self.timer.time("done transform")

        return col1, col2, transformer.fit_transform(document_term_matrix)

    @staticmethod
    def create_word_list(df: pd.DataFrame, col1: str, col2: str) -> List[str]:
        col1_size = df[col1].max() + 1
        col2_list = [[] for _ in range(col1_size)]
        for val1, val2 in zip(df[col1], df[col2]):
            col2_list[val1].append(val2)

        return [' '.join(map(str, list)) for list in col2_list]