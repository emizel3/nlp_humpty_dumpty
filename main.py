
#CSC 4101 Fall 2018
#Authors: Emily Mizell and Michael White
#main
import string
from random import randint
from numpy import array
from pickle import dump
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Embedding
from keras.models import load_model

#load humpty dumpty poem via file
#function for reading a file
def load_doc(name):
    doc = open(name, 'r')
    name_text = doc.read()
    doc.close()
    return name_text
#method for removing punctuation, splitting words, and normalizing text
def cleanup(doc):
    #split
    words = doc.split()
    #remove punctuation
    table = str.maketrans('','',string.punctuation)
    words = [w.translate(table) for w in words]
    #normalize
    words = [word.lower() for word in words]
    return words
def store_file(sentences, name):
    info = '\n'.join(sentences)
    doc = open(name, 'w')
    doc.write(info)
    doc.close()

#read humpty dumpty poem
input = 'humpty.txt'
file = load_doc(input)
#unnecessary print statement - for confirmation only
print(file)
words = cleanup(file)
#unnecessary print statement - for confirmation only
print(words)
#unnecessary print statement - for numerical methods only
print('Total Words: %d' %len(words))
print('Unique Words: %d' % len(set(words)))
#sequence generation
size_seqs = 18
seqs = list()
for i in range(size_seqs, len(words)):
    sequence = words[i-size_seqs:i+1]
    sentence = ' '.join(sequence)
    seqs.append(sentence)
#unnecessary print statement - for confirmation only
print('Total Sequences: %d' % len(seqs))
#save generated sequences
output = 'humpty_seqs.txt'
store_file(seqs, output)
seq_doc = load_doc(output)
sentences = seq_doc.split('\n')
#generate number references for sequences of words
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)
seqs = tokenizer.texts_to_sequences(sentences)
#size of corpus
size_corpus = len(tokenizer.word_index)+1
#input vs output
seqs = array(seqs)
X , y = seqs[:,:-1], seqs[:,-1]
y = to_categorical(y, num_classes = size_corpus)
size_sequence = X.shape[1]
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

size_sequence = len(sentences[0].split()) -1
model = load_model('model.h5')
tokenizer = open('tokenizer.pkl','rb')

def output_sequence(text, n_words):
    seed_text=text
    out = list()
    tokenizer = Tokenizer()
    #unnecessary print statement - for confirmation only
    print(seed_text + '\n')
    #integer encoding
    for _ in range(n_words):
        tokenizer.fit_on_texts([seed_text])[0]
        position = tokenizer.texts_to_sequences([seed_text])[0]
        position = pad_sequences([position], maxlen=size_sequence, truncating='pre')
        yhat = model.predict_classes(position, verbose=0)
        pred_word = ''
        for word, index in tokenizer.word_index.items():
            if index == yhat:
                pred_word = word
                break
        seed_text += ' ' + pred_word
        out.append(pred_word)
    return ' '.join(out)
xx = input('Enter Your Humpty Dumpty Words: ')
gen_output = output_sequence(xx, 1)