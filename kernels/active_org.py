import pandas as pd
import numpy as np
import gc
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import string
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk
from nltk.corpus import stopwords
import scipy
import lightgbm as lgb

sns.set()

used_cols = ['item_id', 'user_id']

train = pd.read_csv('../input/train.csv', usecols=used_cols)
train_active = pd.read_csv('../input/train_active.csv', usecols=used_cols)
test = pd.read_csv('../input/test.csv', usecols=used_cols)
test_active = pd.read_csv('../input/test_active.csv', usecols=used_cols)

train_periods = pd.read_csv('../input/periods_train.csv', parse_dates=['date_from', 'date_to'])
test_periods = pd.read_csv('../input/periods_test.csv', parse_dates=['date_from', 'date_to'])

train.head()

all_samples = pd.concat([
    train,
    train_active,
    test,
    test_active
]).reset_index(drop=True)
all_samples.drop_duplicates(['item_id'], inplace=True)

del train_active
del test_active
gc.collect()


all_periods = pd.concat([
    train_periods,
    test_periods
])

del train_periods
del test_periods
gc.collect()

all_periods.head()

all_periods['days_up'] = all_periods['date_to'].dt.dayofyear - all_periods['date_from'].dt.dayofyear


gp = all_periods.groupby(['item_id'])[['days_up']]

gp_df = pd.DataFrame()
gp_df['days_up_sum'] = gp.sum()['days_up']
gp_df['times_put_up'] = gp.count()['days_up']
gp_df.reset_index(inplace=True)
gp_df.rename(index=str, columns={'index': 'item_id'})

gp_df.head()

all_periods.drop_duplicates(['item_id'], inplace=True)
all_periods = all_periods.merge(gp_df, on='item_id', how='left')
all_periods.head()

all_periods = all_periods.merge(all_samples, on='item_id', how='left')
all_periods.head()

gp = all_periods.groupby(['user_id'])[['days_up_sum', 'times_put_up']].mean().reset_index() \
    .rename(index=str, columns={
        'days_up_sum': 'avg_days_up_user',
        'times_put_up': 'avg_times_up_user'
    })
gp.head()

n_user_items = all_samples.groupby(['user_id'])[['item_id']].count().reset_index() \
    .rename(index=str, columns={
        'item_id': 'n_user_items'
    })
gp = gp.merge(n_user_items, on='user_id', how='outer')

gp.head()

gp.to_csv('aggregated_features.csv', index=False)

del all_samples
del all_periods
del train
del test

gc.collect()

print("-"*40)



train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')

train = train.merge(gp, on='user_id', how='left')
test = test.merge(gp, on='user_id', how='left')

agg_cols = list(gp.columns)[1:]

del gp
gc.collect()

train.head()

train[agg_cols].isna().any(axis=1).sum() / len(train) * 100
test[agg_cols].isna().any(axis=1).sum() / len(test) * 100




