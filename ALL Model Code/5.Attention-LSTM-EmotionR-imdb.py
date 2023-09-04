import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


#Loading the dataset
#dataset = pd.read_csv("emotion.data")
#dataset = pd.read_csv("emotion1k.data")
dataset = pd.read_csv("IMDB Dataset.csv")


# Select data with fixed number of data sample
#X, y = make_blobs(n_samples=1000)
#dataset = dataset(n_samples=1000)

# Plot label histogram
#print("Emotion label visualization from the dataset")
#dataset.sentiment.value_counts().plot.bar()
print("Let predict with model developement")
# Prin some samples
dataset.head(10)
input_sentences = [text.split(" ") for text in dataset["review"].values.tolist()]
labels = dataset["sentiment"].values.tolist()
# Initialize word2id and label2id dictionaries that will be used to encode words and labels
word2id = dict()
label2id = dict()

max_words = 0 # maximum number of words in a sentence

# Construction of word2id dict
for sentence in input_sentences:
    for word in sentence:
        # Add words to word2id dict if not exist
        if word not in word2id:
            word2id[word] = len(word2id)
    # If length of the sentence is greater than max_words, update max_words
    if len(sentence) > max_words:
        max_words = len(sentence)
    
# Construction of label2id and id2label dicts
label2id = {l: i for i, l in enumerate(set(labels))}
id2label = {v: k for k, v in label2id.items()}
print(id2label)

import keras

# Encode input words and labels
X = [[word2id[word] for word in sentence] for sentence in input_sentences]
Y = [label2id[label] for label in labels]

# Apply Padding to X
from keras.preprocessing.sequence import pad_sequences
X = pad_sequences(X, max_words)

# Convert Y to numpy array
Y = keras.utils.to_categorical(Y, num_classes=len(label2id), dtype='float32')

# Print shapes
print("Shape of X: {}".format(X.shape))
print("Shape of Y: {}".format(Y.shape))

#Build LSTM model with attention

embedding_dim = 100 # The dimension of word embeddings

# Define input tensor
sequence_input = keras.Input(shape=(max_words,), dtype='int32')

# Word embedding layer
embedded_inputs =keras.layers.Embedding(len(word2id) + 1,
                                        embedding_dim,
                                        input_length=max_words)(sequence_input)

# Apply dropout to prevent overfitting
embedded_inputs = keras.layers.Dropout(0.2)(embedded_inputs)

#1. LSTM Apply Bidirectional LSTM over embedded inputs
#lstm_outs = keras.layers.wrappers.Bidirectional(keras.layers.LSTM(embedding_dim, return_sequences=True))(embedded_inputs)

#2. BiGRU Apply Bidirectional GRU over embedded inputs
lstm_outs = keras.layers.wrappers.Bidirectional(keras.layers.GRU(embedding_dim, return_sequences=True))(embedded_inputs)

#3. LSTM
#lstm_outs = keras.layers.LSTM(embedding_dim, return_sequences=True)(embedded_inputs)

#4. GRU
#lstm_outs = keras.layers.GRU(embedding_dim, return_sequences=True)(embedded_inputs)

#5. CNN
#lstm_outs = keras.layers.CNN(embedding_dim, return_sequences=True)(embedded_inputs)
#https://github.com/diegoschapira/CNN-Text-Classifier-using-Keras/blob/master/CNN%20Text%20Classifier%20with%20Keras.ipynb
#from keras.models import Sequential
#from keras.layers import Dense, Activation
#from keras.preprocessing.text import Tokenizer, one_hot
#from keras.preprocessing.sequence import pad_sequences
#import sklearn
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
#import matplotlib.pyplot as plt

#vocab_size = 100
#lstm_outs = Sequential()(lstm_outs)
#model_with_attentions  = model.add(layers.Conv2D(em)(embedded_inputs)
#attention_vec = model.add(layers.Conv2D(embedding_dim)(embedded_inputs)
#lstm_outs = model.add(layers.Embedding(vocab_size, embedding_dim, input_length=max_words))(lstm_outs)
#lstm_outs = model.add(layers.Conv1D(128, 5, activation='relu'))(lstm_outs)
#lstm_outs = model.add(layers.GlobalMaxPooling1D())(lstm_outs)
#lstm_outs = model.add(layers.Dense(10, activation='relu'))(lstm_outs)
#attention_vec = model.add(layers.Dense(1, activation='sigmoid'))(lstm_outs)
#model = model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# Apply dropout to LSTM outputs to prevent overfitting
lstm_outs = keras.layers.Dropout(0.2)(lstm_outs)

# Attention Mechanism - Generate attention vectors
input_dim = int(lstm_outs.shape[2])
permuted_inputs = keras.layers.Permute((2, 1))(lstm_outs)
attention_vector = keras.layers.TimeDistributed(keras.layers.Dense(1))(lstm_outs)
attention_vector = keras.layers.Reshape((max_words,))(attention_vector)
attention_vector = keras.layers.Activation('softmax', name='attention_vec')(attention_vector)
attention_output = keras.layers.Dot(axes=1)([lstm_outs, attention_vector])

# Last layer: fully connected with softmax activation
fc = keras.layers.Dense(embedding_dim, activation='relu')(attention_output)
output = keras.layers.Dense(len(label2id), activation='softmax')(fc)

# Finally building model

model = keras.Model(inputs=[sequence_input], outputs=output)
#model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer='adam')
# Print model summary
#model.summary()

# Train model 10 iterations
# I changed epoches from two to one
#model.fit(X, Y, epochs=1, batch_size=64, validation_split=0.95, shuffle=True)


#'''
########################### VISUALIZATION OF ACCURACY way 01 ################################################################## 
from sklearn.model_selection import train_test_split
#model.fit(X, Y, epochs=1, batch_size=64, train_size=0.95, shuffle=True)
#x_train, x_test, y_train, y_test = train_test_split(X, Y, train_size= 0.50, random_state=1000)
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
#history = model.fit(x_test, y_test, validation_split=0.50, shuffle=True)
history=model.fit(X, Y, epochs=3, batch_size=256, validation_split=0.05, shuffle=True)
#validations  splite means (value 0.95 means training is .05 and testing for .95 )

#model.test_on_batch(x_test, y_test)
model.metrics_names
#I have plotted accuracy and loss of training and validation:
import matplotlib.pyplot as plt
print(history.history.keys())
#  "Accuracy"
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Attention based capsule network with Bidirectional GRU Model accuracy measurement')
plt.ylabel('Accuracy')
plt.xlabel('Number of epoches')
plt.legend(['Training Accuracy', 'Testing or validation accuracy'], loc='upper left')
plt.show()
# "Loss"
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Attention based capsule network with GRU Model model loss')
plt.ylabel('Loss')
plt.xlabel('Number of epoches')
plt.legend(['Traing Loss', 'Testing or validation Loss'], loc='upper left')
plt.show()
#'''

#Let's look closer to model predictions and attentions

# Re-create the model to get attention vectors as well as label prediction
model_with_attentions = keras.Model(inputs=model.input, outputs=[model.output, model.get_layer('attention_vec').output])
    

# VISUALIZATION OF TRAINING AND TESTING ACCURACY 02
##from sklearn.cross_validation import train_test_split

#from sklearn.model_selection import train_test_split
#from sklearn.feature_extraction.text import CountVectorizer
##history = model.fit(X_train, y_train, epochs=10, verbose=False, validation_data=(X_test, y_test), batch_size=32)
##history = model.fit(X, Y, epochs=1, batch_size=64, validation_split=0.8, shuffle=True)
##X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size= 0.10, test_size=0.25, random_state=1000)
#X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size= 0.10, random_state=1000)
## TRY WITH TRAINING SIZE IF POSSIBLE

##vectorizer = CountVectorizer()
##vectorizer.fit(sentences_train)

##X_train = vectorizer.transform(sentences_train)
##X_test  = vectorizer.transform(sentences_test)
'''
historyTrain = model.fit(X_train, y_train, epochs=10, verbose=False, validation_data=(X_test, y_test), batch_size=32)
historyTest = model.fit(X_test, y_test, epochs=10, verbose=False, validation_data=(X_test, y_test), batch_size=32)

#history = model_with_attentions = keras.Model(inputs=model.input, outputs=[model.output, model.get_layer('attention_vec').output])

loss, accuracy = model.evaluate(X_train, y_train, verbose=False)
print("Training Accuracy: {:.4f}".format(accuracy))
loss, accuracy = model.evaluate(X_test, y_test, verbose=False)
print("Testing Accuracy:  {:.4f}".format(accuracy))
print("Train graph")
plot_history(historyTrain)
print("Train graph")
plot_history(historyTest)
'''

'''
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

    x = Bidirectional(
        GRU(gru_len, activation='relu', dropout=dropout_p, recurrent_dropout=dropout_p, return_sequences=True))(
        embed_layer)
    capsule = Capsule(num_capsule=Num_capsule, dim_capsule=Dim_capsule, routings=Routings,
                      share_weights=True)(x)
    # output_capsule = Lambda(lambda x: K.sqrt(K.sum(K.square(x), 2)))(capsule)
    attention = Flatten()(capsule)
    attention = Dropout(dropout_p)(capsule)
    output = Dense(6, activation='sigmoid')(attention)
    model = Model(inputs=input1, outputs=output)
    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])
    model.summary()
    return model
model = get_model()
'''

#################### VisuALIZATION

import random
import math

# Select random samples to illustrate
#I given sentence manually but it should be in lowercase letter
#sample_text = random.choice(dataset["text"].values.tolist())
#sample_text = "i loved feeling festive all day"
#sample_text = "i love research and feeling good when write code but i dont like more exam"
print("Emotion visualization for input text")
sample_text = "i love love romantic movie but i do not like other movie"

# Encode samples
tokenized_sample = sample_text.split(" ")
encoded_samples = [[word2id[word] for word in tokenized_sample]]

# Padding
encoded_samples = keras.preprocessing.sequence.pad_sequences(encoded_samples, maxlen=max_words)

# Make predictions
label_probs, attentions = model_with_attentions.predict(encoded_samples)
label_probs = {id2label[_id]: prob for (label, _id), prob in zip(label2id.items(),label_probs[0])}

# Get word attentions using attenion vector
token_attention_dic = {}
max_score = 0.0
min_score = 0.0
for token, attention_score in zip(tokenized_sample, attentions[0][-len(tokenized_sample):]):
    token_attention_dic[token] = math.sqrt(attention_score)


# VISUALIZATION
import matplotlib.pyplot as plt; plt.rcdefaults()
#import numpy as np
import matplotlib.pyplot as plt
from IPython.core.display import display, HTML

def rgb_to_hex(rgb):
    return '#%02x%02x%02x' % rgb
    
def attention2color(attention_score):
    r = 255 - int(attention_score * 255)
    color = rgb_to_hex((255, r, r))
    return str(color)
    
# Build HTML String to viualize attentions
html_text = "<hr><p style='font-size: large'><b>Text:  </b>"
for token, attention in token_attention_dic.items():
    html_text += "<span style='background-color:{};'>{} <span> ".format(attention2color(attention),
                                                                        token)
html_text += "</p>"
# Display text enriched with attention scores 
display(HTML(html_text))

# PLOT EMOTION SCORES
sentiment = [label for label, _ in label_probs.items()]
scores = [score for _, score in label_probs.items()]
plt.figure(figsize=(5,2))
#plt.bar(np.arange(len(sentiment)), scores, align='center', alpha=0.5, color=['black', 'red', 'green', 'blue', 'cyan', "purple"])
plt.bar(np.arange(len(sentiment)), scores, align='center', alpha=0.5, color=['green', 'red'])

plt.xticks(np.arange(len(sentiment)), sentiment)
plt.ylabel('Scores')
plt.show()

print("Emotion label visualization from the dataset")
dataset.sentiment.value_counts().plot.bar()
