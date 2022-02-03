"""
From https://machinelearningmastery.com/develop-word-based-neural-language-models-python-keras/
"""
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from numpy import array
import random
import numpy as np
from lingpy.sequence.sound_classes import get_all_ngrams
from lingpy import Wordlist

## now add English data form WORLD or similar TODO

# check also here: https://datascience.stackexchange.com/questions/26366/training-an-rnn-with-examples-of-different-lengths-in-keras

 
# generate a sequence from the model
def generate_seq(model, tokenizer, seed_text, n_words):
    # generate a fixed number of words
    result = seed_text.split()
    current_text = seed_text
    for _ in range(n_words):
        # encode the text as integer
        encoded = tokenizer.texts_to_sequences([current_text])[0]
        pref = 20 * [0]
        pref += encoded
        encoded = array([pref[-20:]])
        # predict a word in the vocabulary
        
        predict_x=model.predict(encoded) 
        yhat = np.argmax(predict_x, axis=1)[0]
        yhat = np.random.choice(len(predict_x[0]), p=predict_x[0])
        out_word = ''
        res = tokenizer.idx2tok.get(yhat, "$")
        if res == "$":
            break
        result += [res]
        current_text = " ".join(result[-2:])

    return result
 
# source text
data = """w e r $ w i e $ w a s $ d e r $ d i e $ d a s $ w i e s o $ w e s h a l b $ w a r u m $ w e r $ n i c h t $ f r a g t $ b l e i b t $ d u m m""".split(" $ ")

wl = Wordlist.from_cldf("wold/cldf/cldf-metadata.json")
english = wl.get_list(col="English", entry="tokens", flat=True)
data = ["START "+" ".join(x) for x in english]
    
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data)
tokenizer.idx2tok = {idx: w for w, idx in tokenizer.word_index.items()}
encoded = tokenizer.texts_to_sequences(data)

# determine the vocabulary size
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)
# create word -> word sequences
sequences = list()
for enco in encoded:
    for i in range(5):
        ng = random.choice([x for x in get_all_ngrams(enco+[0]) if len(x) > 1])
        if ng:
            nng = 21 * [0]
            nng += ng
            ngn = nng[-21:]
            sequences.append(ngn)
print('Total Sequences: %d' % len(sequences))
# split into X and y elements
sequencesx = array(sequences)
X, y = sequencesx[:, 0:20], sequencesx[:, 20]
# one hot encode outputs
y = to_categorical(y, num_classes=vocab_size)
# define model
model = Sequential()
model.add(Embedding(vocab_size, 10, input_length=20, mask_zero=True))
model.add(LSTM(50))
model.add(Dense(vocab_size, activation='softmax'))
print(model.summary())
# compile network
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit network
model.fit(X, y, epochs=100, verbose=1)
# evaluate
for w in "ptk":
    for x in range(10):
        pred = " ".join(generate_seq(model, tokenizer, "START", 10)[1:])
        print(pred)
    print("")
