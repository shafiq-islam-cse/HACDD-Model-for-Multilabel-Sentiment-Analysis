# Developer Shofiqul Islam, UMP malaysia.
# data from longnguyen2306 at kaggle with capsule-net-with-gru
#y_train = train[["joy", "anger", "surprise", "love", "sadness", "fear"]].values
#submission[["joy", "anger", "surprise", "love", "sadness", "fear"]] = y_pred
# Developer Shofiqul Islam, UMP malaysia.
# data from longnguyen2306 at kaggle with capsule-net-with-gru

# Developer Shofiqul Islam, UMP malaysia.
# data from longnguyen2306 at kaggle with capsule-net-with-gru

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os
from keras.layers import Dense, Input, LSTM, Bidirectional, Activation, Conv1D, GRU
from keras.callbacks import Callback
from keras.layers import Dropout, Embedding, GlobalMaxPooling1D, MaxPooling1D, Add, Flatten
from keras.preprocessing import text, sequence
from keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, concatenate, SpatialDropout1D
from keras import initializers, regularizers, constraints, optimizers, layers, callbacks
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score

# print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.
#EMBEDDING_FILE = 'glove.840B.300d.txt'
EMBEDDING_FILE = 'D:/My Spyder/0 toxic comments classification/glove.840B.300d.txt'
# train = pd.read_csv('../input/cleaned-toxic-comments/train_preprocessed.csv')
train = pd.read_csv('IMDBmodified.csv')
y_train = train[["positive", "negative"]].values

#y_train = train[["joy", "anger", "surprise", "love", "sadness", "fear"]].values
# test = pd.read_csv('../input/cleaned-toxic-comments/test_preprocessed.csv')
test = pd.read_csv('IMDBmodified.csv')
train["review"].fillna("fillna")
test["review"].fillna("fillna")
X_train = train["review"].str.lower()
#y_train = train[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
#y_train = train[["joy", "anger", "surprise", "love", "sadness", "fear"]].values
y_train = train[["positive", "negative"]].values
X_test = test["review"].str.lower()

max_features = 100000
#max_features = 75303
maxlen = 500
embed_size = 300


class RocAucEvaluation(Callback):
    def __init__(self, validation_data=(), interval=1):
        super(Callback, self).__init__()

        self.interval = interval
        self.X_val, self.y_val = validation_data

    def on_epoch_end(self, epoch, logs={}):
        if epoch % self.interval == 0:
            y_pred = self.model.predict(self.X_val, verbose=0)
            score = roc_auc_score(self.y_val, y_pred)
            print("\n ROC-AUC - epoch: {:d} - score: {:.6f}".format(epoch + 1, score))


tok = text.Tokenizer(num_words=max_features, lower=True)
tok.fit_on_texts(list(X_train) + list(X_test))
X_train = tok.texts_to_sequences(X_train)
X_test = tok.texts_to_sequences(X_test)
x_train = sequence.pad_sequences(X_train, maxlen=maxlen)
x_test = sequence.pad_sequences(X_test, maxlen=maxlen)

embeddings_index = {}
with open(EMBEDDING_FILE, encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
word_index = tok.word_index
# prepare embedding matrix
num_words = min(max_features, len(word_index) + 1)
embedding_matrix = np.zeros((num_words, embed_size))
for word, i in word_index.items():
    if i >= max_features:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

from keras.layers import Activation  # Added this
# from tensorflow.python.keras.layers import K, Activation
from keras import backend as K
# from tensorflow.keras.engine import Layer#                                                           Changed here
from keras.layers import Layer  # Added this
from keras.layers import Dense, Input, Embedding, Dropout, Bidirectional, GRU, Flatten, SpatialDropout1D

gru_len = 128
Routings = 5
Num_capsule = 10
Dim_capsule = 16
dropout_p = 0.25
rate_drop_dense = 0.28


def squash(x, axis=-1):
    # s_squared_norm is really small
    # s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
    # scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
    # return scale * x
    s_squared_norm = K.sum(K.square(x), axis, keepdims=True)
    scale = K.sqrt(s_squared_norm + K.epsilon())
    return x / scale


# A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, kernel_size=(9, 1), share_weights=True,
                 activation='default', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.kernel_size = kernel_size
        self.share_weights = share_weights
        if activation == 'default':  # Changed 3 lines here
            self.activation = squash
        else:
            self.activation = Activation(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     # shape=self.kernel_size,
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])

        batch_size = K.shape(u_vecs)[0]
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        # final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:, :, :, 0])  # shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            b = K.permute_dimensions(b, (0, 2, 1))  # shape = [None, input_num_capsule, num_capsule]
            c = K.softmax(b)
            c = K.permute_dimensions(c, (0, 2, 1))
            b = K.permute_dimensions(b, (0, 2, 1))
            outputs = self.activation(K.batch_dot(c, u_hat_vecs, [2, 2]))
            if i < self.routings - 1:
                b = K.batch_dot(outputs, u_hat_vecs, [2, 3])

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)


def get_model():
    input1 = Input(shape=(maxlen,))
    embed_layer = Embedding(max_features,
                            embed_size,
                            input_length=maxlen,
                            weights=[embedding_matrix],
                            trainable=False)(input1)
    embed_layer = SpatialDropout1D(rate_drop_dense)(embed_layer)
'''
    x = Bidirectional(
        GRU(gru_len, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, return_sequences=True))(
        embed_layer)
    x = Bidirectional(
        GRU(gru_len, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, return_sequences=True))(x)
'''
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                      share_weights=True)(embed_layer)
    # output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)
    capsule = Flatten()(capsule)
    capsule = Dropout(dropout_p)(capsule)
    output = Dense(2, activation='sigmoid')(capsule)# Dense function value changes with number of class in the dataset to be trained
    model = Model(inputs=input1, outputs=output)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()
    return model
#loss='binary_crossentropy',
#loss='categorical_crossentropy'
model = get_model()

batch_size = 256  # Changed 256 to 128
epochs = 4
X_tra, X_val, y_tra, y_val = train_test_split(x_train, y_train, train_size=0.80,
                                              random_state=233)  # Changed train size from 0.95 to .30
RocAuc = RocAucEvaluation(validation_data=(X_val, y_val), interval=1)
hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=1, validation_data=(X_val, y_val),
                 callbacks=[RocAuc], verbose=1)
'''
hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=1, validation_data=(X_val, y_val),
                 callbacks=[RocAuc], verbose=1)
hist = model.fit(X_tra, y_tra, batch_size=batch_size, epochs=1, validation_data=(X_val, y_val),
                 callbacks=[RocAuc], verbose=1)
#sample="I am not good for you"
#X-Test=sample.str.lower()
#y_pred = model.predict("I love you", batch_size=1024, verbose=1)
#y_pred = model.predict(x-test, batch_size=1024, verbose=1)
#print("Prediction value for input text")
#print(y_pred)
y_pred = model.predict(x_test, batch_size=1024, verbose=1)
#submission = pd.read_csv('sample_submission.csv')
#submission[["joy", "anger", "surprise", "love", "sadness", "fear"]] = y_pred
#submission[["positive", "negative"]] = y_pred
#submission[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]] = y_pred
#submission.to_csv('submission.csv', index=False)
model.save_weights('IMDBbest.hdf5')
#shape placeholder learning_rate
'''
