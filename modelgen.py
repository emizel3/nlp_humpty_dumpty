import string
from numpy import array
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding

#generate model
model = Sequential()
model.add(Embedding(size_corpus, 18, input_length = size_seqs))
model.add(LSTM(100, return_sequences = True))
model.add(LSTM(100))
model.add(Dense(100, activation='relu'))
model.add(Dense(size_corpus, activation='softmax'))
#unnecessary print statement - for confirmation only
print(model.summary())
#compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, batch_size=100, epochs=100)
#save
model.save('model.h5')
dump(tokenizer, open('tokenizer.pkl', 'wb'))