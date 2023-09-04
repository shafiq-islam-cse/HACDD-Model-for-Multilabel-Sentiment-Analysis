# Bangla text analysis is at https://www.kaggle.com/nuhashafnan/bangla-sentiment-analysis-cnn-lstm-hybrid-network
import pandas as pd
from pandas import read_excel
import numpy as np
import re
from re import sub
import multiprocessing
#from unidecode import unidecode
import os
from time import time 
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import LSTM,Dense,Dropout,Activation,Embedding,Flatten,Bidirectional,MaxPooling2D, Conv1D, MaxPooling1D
from keras.optimizers import SGD,Adam
from keras import regularizers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
import h5py
import csv
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

def text_to_word_list(text):
    text = text.split()
    return text


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#{0: 'empty', 1: 'worry', 2: 'anger', 3: 'hate', 4: 'sadness', 5: 'enthusiasm', 6: 'happiness',
# 7: 'boredom', 8: 'neutral', 9: 'fun', 10: 'surprise', 11: 'relief', 12: 'love'}
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
Y = keras.utils.to_categorical(Y, num_classes=len(label2id))

# Print shapes
print("Shape of X: {}".format(X.shape))
print("Shape of Y: {}".format(Y.shape))

'''
def replace_strings(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           u"\u00C0-\u017F"          #latin
                           u"\u2000-\u206F"          #generalPunctuations
                               
                           "]+", flags=re.UNICODE)
    english_pattern=re.compile('[a-zA-Z0-9]+', flags=re.I)
    #latin_pattern=re.compile('[A-Za-z\u00C0-\u00D6\u00D8-\u00f6\u00f8-\u00ff\s]*',)
    
    text=emoji_pattern.sub(r'', text)
    text=english_pattern.sub(r'', text)

    return text

def remove_punctuations(my_str):
    # define punctuation
    punctuations = ````£|¢|Ñ+-*/=EROero???????????012–34567•89?!()-[]{};:'"“\’,<>./?@#$%^&*_~‘—?”‰???????
    
    no_punct = ""
    for char in my_str:
        if char not in punctuations:
            no_punct = no_punct + char

    # display the unpunctuated string
    return no_punct

def joining(text):
    out=' '.join(text)
    return out

def preprocessing(text):
    out=remove_punctuations(replace_strings(text))
    return out
'''
'''

#                                                       DATA INPUT HERE #################################################################
#df=pd.read_excel('/kaggle/input/pseudolabel/predicted_unsupervised_sentiment.xlsx')
#df = pd.read_csv('Crowdflower eomotion recognition dataset.csv')
#df = pd.read_csv('IMDB Dataset.csv')
df = pd.read_csv('IMDBmodified.csv', usecols=['review','sentiment'])

#display(df)
	
#sns.countplot(df['sentiment']);

#df['sentence'] = df.sentence.apply(lambda x: preprocessing(str(x)))
#df.reset_index(drop=True, inplace=True)

train1, test1 = train_test_split(df,random_state=69, test_size=0.2)
training_sentences = []
testing_sentences = []
#train_sentences=train1['content'].values
#train_labels=train1['sentiment'].values


test_sentences=test1['review'].values
test_labels=test1['sentiment'].values

#train_sentences=train1['sentence'].values
#train_labels=train1['sentiment'].values
for i in range(train_sentences.shape[0]): 
    #print(train_sentences[i])
    x=str(train_sentences[i])
    training_sentences.append(x)
    
training_sentences=np.array(training_sentences)

# DATA ###################################################################
test_sentences=test1['review'].values
test_labels=test1['sentiment'].values
#########################################################################

#y_train = np.asarray(train_labels).astype('float32').reshape((-1,1)) 
#y_test = np.asarray(test_labels).astype('float32').reshape((-1,1))

#train_labels = np.asarray(train_labels).astype('float32').reshape((-1,1)) 
#test_labels = np.asarray(test_labels).astype('float32').reshape((-1,1))

for i in range(test_sentences.shape[0]): 
    x=str(test_sentences[i])
    testing_sentences.append(x)
    
testing_sentences=np.array(testing_sentences)


#train_labels=keras.utils.to_categorical(train_labels)


#test_labels=keras.utils.to_categorical(test_labels)

print("PRINTING SHAPE OF INPUT")

print("Training Set Length: "+str(len(train1)))
print("Testing Set Length: "+str(len(test1)))
print("training_sentences shape: "+str(training_sentences.shape))
print("testing_sentences shape: "+str(testing_sentences.shape))
print("train_labels shape: "+str(train_labels.shape))
print("test_labels shape: "+str(test_labels.shape))

print(training_sentences[1])
print(train_labels[0])
'''
# SETTING PARAMETERS
vocab_size = 750000
embedding_dim = 300
max_length = 100
trunc_type='post'
oov_tok = "<OOV>"
'''
print(training_sentences.shape)
print(train_labels.shape)

tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)
word_index = tokenizer.word_index
print(len(word_index))
print("Word index length:"+str(len(tokenizer.word_index)))

# PAD SEQUENCE
sequences = tokenizer.texts_to_sequences(training_sentences)
padded = pad_sequences(sequences,maxlen=max_length, truncating=trunc_type)


test_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = pad_sequences(test_sequences,maxlen=max_length)
'''
'''
print("Sentence :--> \n")
print(training_sentences[2]+"\n")
print("Sentence Tokenized and Converted into Sequence :--> \n")
print(str(sequences[2])+"\n")
print("After Padding the Sequence with padding length 100 :--> \n")
print(padded[2])

print("Padded shape(training): "+str(padded.shape))
print("Padded shape(testing): "+str(testing_padded.shape))
'''
#                                  CNN AND LSTM
with tf.device('/gpu:0'):
    model= Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
    model.add(Conv1D(200, kernel_size=3, activation = "relu"))
    model.add(Bidirectional(LSTM(64, return_sequences=True)))
    model.add(Dropout(0.5))
    model.add(Bidirectional(LSTM(64)))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    #model.add(Flatten())
    #l2 regularizer
    model.add(Dense(100,kernel_regularizer=regularizers.l2(0.01),activation="relu"))
    model.add(Dense(2, activation='softmax'))
    #sgd= SGD(lr=0.0001,decay=1e-6,momentum=0.9,nesterov=True)
    #adam=Adam(learning_rate=0.0005,beta_1=0.9,beta_2=0.999,epsilon=1e-07,amsgrad=False)
    model.summary()
    #model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])
    #model.compile(loss='binary_crossentropy',optimizer=adam,metrics=['accuracy'])

#history=model.fit(padded,train_labels,epochs=5,batch_size=256,validation_data=(testing_padded,test_labels),use_multiprocessing=True, workers=8)

from sklearn.model_selection import train_test_split

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


X_tra, X_val, y_tra, y_val = train_test_split(X, Y, train_size=0.80, test_size=0.20,
                                              random_state=233)  # Changed train size from 0.95 to .30
history = model.fit(X_tra, y_tra, batch_size=32, epochs=4, validation_data=(X_val, y_val), verbose=1)


Visualization
print(history.history.keys())
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(loss)
plt.plot(val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'])
plt.show()

accuracy = history.history['accuracy']
val_accuracy= history.history['val_accuracy']
plt.plot(accuracy)
plt.plot(val_accuracy)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy', 'val_accuracy'])
plt.show()

Accuracy and Evaluation
#accuracy calculation
loss_and_metrics = model.evaluate(padded,train_labels,batch_size=256)
print("The train accuracy is: "+str(loss_and_metrics[1]))
loss_and_metrics = model.evaluate(testing_padded,test_labels,batch_size=256)
print("The test accuracy is: "+str(loss_and_metrics[1]))


'''
#                                  CNN 
with tf.device('/gpu:0'):
    model= Sequential()
    model.add(Embedding(vocab_size, embedding_dim, input_length=max_length))
    model.add(Conv1D(200, kernel_size=3, activation = "relu"))
    #model.add(Bidirectional(LSTM(64, return_sequences=True)))
    #model.add(Dropout(0.5))
    #model.add(Bidirectional(LSTM(64)))
    #model.add(Dense(50, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Flatten())
    #l2 regularizer
    model.add(Dense(100,kernel_regularizer=regularizers.l2(0.01),activation="relu"))
    model.add(Dense(2, activation='softmax'))
    #sgd= SGD(lr=0.0001,decay=1e-6,momentum=0.9,nesterov=True)
    adam=Adam(learning_rate=0.0005,beta_1=0.9,beta_2=0.999,epsilon=1e-07,amsgrad=False)
    model.summary()
    model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])

 history=model.fit(padded,train_labels,epochs=5,batch_size=256,validation_data=(testing_padded,test_labels),use_multiprocessing=True, workers=8)


Visualization
print(history.history.keys())
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(loss)
plt.plot(val_loss)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'val_loss'])
plt.show()

accuracy = history.history['accuracy']
val_accuracy= history.history['val_accuracy']
plt.plot(accuracy)
plt.plot(val_accuracy)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy', 'val_accuracy'])
plt.show()

Accuracy and Evaluation
#accuracy calculation
loss_and_metrics = model.evaluate(padded,train_labels,batch_size=256)
print("The train accuracy is: "+str(loss_and_metrics[1]))
loss_and_metrics = model.evaluate(testing_padded,test_labels,batch_size=256)
print("The test accuracy is: "+str(loss_and_metrics[1]))
'''