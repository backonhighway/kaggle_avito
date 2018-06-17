import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

train = pd.DataFrame({
    "user_id": [3,4,3,3,4,4,5,5,4,3],
    "city": [10,10,3,3,3,10,4,10,10,10],
})

# city, user_id, param_all, image_top_1
def create_word_list(df: pd.DataFrame, col1: str, col2: str):
    col1_size = df[col1].max() + 1
    col2_list = [[] for _ in range(col1_size)]
    for val1, val2 in zip(df[col1], df[col2]):
        col2_list[val1].append(val2)
    return [' '.join(map(str, list)) for list in col2_list]

city_of_user_id = {}
for sample in train:
    print(sample)
    city_of_user_id.setdefault(sample["user_id"], []).append(str(sample["city"]))
print(city_of_user_id)

user_ids = list(city_of_user_id.keys())
print(user_ids)

city_as_sentence = [" ".join(city_of_user_id[user_id]) for user_id in user_ids]
print(city_as_sentence)

city_as_matrix = CountVectorizer().fit_transform(city_as_sentence)
print(city_as_matrix)

lda = LatentDirichletAllocation(n_components=2, learning_method="online", random_state=99)
topic_of_user_id = lda.fit_transform(city_as_matrix)
print(topic_of_user_id)


