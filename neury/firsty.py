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
EMB_FILE = os.path.join(INPUT_DIR, "cc.ru.300.vec")

import pandas as pd
import numpy as np
from keras.preprocessing import text, sequence
from tqdm import tqdm
from keras.layers import Input, SpatialDropout1D,Dropout, GlobalAveragePooling1D, \
    LSTM, GRU, CuDNNGRU, Bidirectional, Dense, Embedding
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from sklearn import metrics
from sklearn import model_selection
from avito.common import pocket_timer, pocket_logger, column_selector

logger = pocket_logger.get_my_logger()
timer = pocket_timer.GoldenTimer(logger)


embed_size = 300

train = pd.read_csv(ORG_TRAIN, nrows=1000*100)
labels = train[['deal_probability']].copy()
description_series = train["description"].fillna("missing")

max_features=1000*100
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(description_series))
tokenized_description = tokenizer.texts_to_sequences(description_series)

maxlen = 100
description_seq = sequence.pad_sequences(tokenized_description, maxlen=maxlen)

# totalNumWords = [len(one_comment) for one_comment in tokenized_description]
# import matplotlib.pyplot as plt
# plt.hist(totalNumWords, bins=np.arange(0, 410, 10))#[0,50,100,150,200,250,300,350,400])#,450,500,550,600,650,700,750,800,850,900])
# plt.show()
timer.time("done tokenize")
train_y = train["deal_probability"]
train_x = description_seq
X_train, X_valid, y_train, y_valid = \
    model_selection.train_test_split(train_x, train_y, test_size=0.1, random_state=99)

def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in tqdm(open(EMB_FILE)))

word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in tqdm(word_index.items()):
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector

del embeddings_index
timer.time("done embeddings")

train_y = train["deal_probability"]
train_x = description_seq
X_train, X_valid, y_train, y_valid = \
    model_selection.train_test_split(train_x, train_y, test_size=0.1, random_state=99)
del train
# print('convert to sequences')
# X_train = tokenizer.texts_to_sequences(X_train)
# X_valid = tokenizer.texts_to_sequences(X_valid)
#
# print('padding')
# X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
# X_valid = sequence.pad_sequences(X_valid, maxlen=maxlen)

def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def rmse(y_true, y_pred):
    return np.sqrt(metrics.mean_squared_error(y_valid, y_pred))

def build_model():
    inp = Input(shape = (maxlen, ))
    emb = Embedding(nb_words, embed_size, weights=[embedding_matrix],
                    input_length=maxlen, trainable=False)(inp)
    main = SpatialDropout1D(0.2)(emb)
    main = Bidirectional(GRU(32,return_sequences = True))(main)
    main = GlobalAveragePooling1D()(main)
    main = Dropout(0.2)(main)
    out = Dense(1, activation="sigmoid")(main)

    model = Model(inputs=inp, outputs=out)

    model.compile(optimizer = Adam(lr=0.001), loss = 'mean_squared_error',
                  metrics =[root_mean_squared_error])
    model.summary()
    return model

EPOCHS = 4

the_model = build_model()
file_path = "model.hdf5"

check_point = ModelCheckpoint(file_path, monitor = "val_loss", mode = "min", save_best_only = True, verbose = 1)
history = the_model.fit(X_train, y_train, batch_size = 256, epochs = EPOCHS, validation_data = (X_valid, y_valid),
                verbose = 1, callbacks = [check_point])
the_model.load_weights(file_path)
prediction = the_model.predict(X_valid)
print('RMSE:', rmse(y_valid, prediction))


exit(0)
test = pd.read_csv(PRED_TEST, index_col = 0)
test = test[['description']].copy()

test['description'] = test['description'].astype(str)
X_test = test['description'].values
X_test = tokenizer.texts_to_sequences(X_test)

print('padding')
X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
prediction = the_model.predict(X_test,batch_size = 128, verbose = 1)

sample_submission = pd.read_csv('../input/avito-demand-prediction/sample_submission.csv', index_col = 0)
submission = sample_submission.copy()
submission['deal_probability'] = prediction
submission.to_csv('submission.csv')



