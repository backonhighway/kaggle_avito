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
        "Бытовая электроника",
        "Бытовая",
        "электроника",
        "Для бизнеса",
        "Для дома и дачи"
    ]
})

russian_caps = "[АБВГДЕЁЖЗИЙКЛМНОПРСТУФХЦЧШЩЪЫЬЭЮЯ]"
dft["count"] = dft["text"].apply(len)
dft["upper"] = dft["text"].str.count(russian_caps)
print(dft)


