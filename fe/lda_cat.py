import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../'))
sys.path.append(ROOT)
APP_ROOT = os.path.join(ROOT, "avito")
INPUT_DIR = os.path.join(APP_ROOT, "input")
OUTPUT_DIR = os.path.join(APP_ROOT, "output")
ORG_TRAIN = os.path.join(INPUT_DIR, "train.csv")
ORG_TEST = os.path.join(INPUT_DIR, "test.csv")
PRED_TRAIN = os.path.join(OUTPUT_DIR, "pred_train.csv")
PRED_TEST = os.path.join(OUTPUT_DIR, "pred_test.csv")
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


train = pd.read_csv(ORG_TRAIN, nrows=1000*100)
#test = pd.read_csv(ORG_TEST)

# city, user_id, param_all, image_top_1

city_of_user_id = {}
for sample in train:
    city_of_user_id.setdefault(sample["user_id"], []).append(str(sample["city"]))
user_ids = list(city_of_user_id.keys())
city_as_sentence = [" ".join(city_of_user_id[user_id]) for user_id in user_ids]
city_as_matrix = CountVectorizer().fit_transform(city_as_sentence)

lda = LatentDirichletAllocation(n_components=5, learning_method="online", random_state=99)
topic_of_user_id = lda.fit_transform(city_as_matrix)