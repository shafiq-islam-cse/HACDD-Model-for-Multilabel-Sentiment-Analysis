# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
'''
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
'''
# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

df=pd.read_csv("IMDB Dataset.csv")
df.head()

import re
from nltk.corpus import stopwords
import string
embedding_index=dict()
f=open('glove.6B.100d.txt',encoding="utf-8")
for line in f:
    values=line.split()
    word=values[0]
    enbd=np.asarray(values[1:],dtype='float32')
    embedding_index[word]=enbd

f.close()
def clean_data(docs):
    review_list=[]
    for review in docs:
        tokens=review.split();
        re_punc=re.compile('[%s]'%re.escape(string.punctuation))
        tokens=[re_punc.sub('',w) for w in tokens]
        tokens=[word for word in tokens if word.isalpha()]
        stop_words=set(stopwords.words('english'))
        tokens=[w for w in tokens if not w in stop_words]
#         tokens=' '.join(tokens)
        review_list.append(tokens)
    return review_list

clean_reviews=clean_data(df['review'].values)


from collections import Counter
vocab=Counter()
for review in clean_reviews:
    vocab.update(review)
def save_list(lines, filename):
    data='\n'.join(lines)
    file=open(filename,'w')
    file.write(data)
    file.close()
tokens=[k for k,c in vocab.items() if c>=5]
print(len(tokens))
save_list(tokens,'vocab.txt')

def clean_doc(docs,vocab):
    review_list=[]
    for review in docs:
        tokens=review.split();
        re_punc=re.compile('[%s]'%re.escape(string.punctuation))
        tokens=[re_punc.sub('',w) for w in tokens]
        tokens=[word for word in tokens if word.isalpha()]
        stop_words=set(stopwords.words('english'))
        tokens=[w for w in tokens if not w in stop_words]
        tokens=[w for w in tokens if w in vocab]
        tokens=' '.join(tokens)
        review_list.append(tokens)
    return review_list
file=open('vocab.txt','r')
vocab=file.read()
clean_reviews=clean_doc(df['review'].values,vocab)
df['clean_reviews']=clean_reviews
df.head()
clean_reviews[0]
from sklearn.model_selection import train_test_split
df['sentiment']=(df['sentiment']=='positive')*1
train_doc,test_doc,y_train,y_test=train_test_split(df['clean_reviews'],df['sentiment'])
y_train.head()
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
def create_tokenizer(lines):
    tokenizer=Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer
def encode_docs(tokenizer,max_length,docs):
    encoded=tokenizer.texts_to_sequences(docs)
    padded=pad_sequences(encoded,maxlen=max_length,padding='post')
    return padded
tokenizer=create_tokenizer(train_doc)
vocab_size=len(tokenizer.word_index)+1
#vocab_size
avg_length=sum([len(s.split()) for s in train_doc])/len(train_doc)
print(avg_length)
max_length=max([len(s.split()) for s in train_doc])
print(max_length)
#from keras.utils.vis_utils import plot_model
from tensorflow.keras.layers import Dense, Flatten,Embedding
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D
import tensorflow as tf
embedding_matrix=np.zeros((vocab_size,100))
for word ,i in tokenizer.word_index.items():
  embedding_vector=embedding_index.get(word)
  if embedding_vector is not None:
    embedding_matrix[i]=embedding_vector
max_length=500
X_train=encode_docs(tokenizer,max_length,train_doc)
X_test=encode_docs(tokenizer,max_length,test_doc)
def define_model(vocab_size,max_length):
    model=Sequential()
    model.add(Embedding(vocab_size,100,weights=[embedding_matrix],input_length=max_length,trainable=False))
    model.add(Conv1D(filters=32,kernel_size=4,activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Conv1D(filters=64,kernel_size=4,activation='relu'))
    model.add(MaxPooling1D(pool_size=2))
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dense(16,activation='relu'))
    model.add(Dense(1,activation='sigmoid'))

    model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
    model.summary()
    #plot_model(model,show_shapes=True)
    return model



model=define_model(vocab_size,max_length)
callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
model.fit(X_train,y_train,epochs=3,validation_split=.2,callbacks=[callback])

model.evaluate(X_test,y_test)


review=["There is serious problem with Netflix India sumit74804 October 2020 There is serious problem with Netflix India, Outside India Netflix is good but In india netflix has some kind of agenda, Important thing One thing they should be clear that they choose actor according to character , it should not be like nawazzudin is good actor so every body should give him every type of character, Just like Radhika Apte, And they should choose some good directors"]

review=encode_docs(tokenizer,max_length,review)
model.predict(review)

history = model.fit(X_train,y_train,epochs=10,validation_split=.2,callbacks=[callback])
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
