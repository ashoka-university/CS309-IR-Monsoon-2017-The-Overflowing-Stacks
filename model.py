#import statements
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM, Bidirectional
from keras.layers.embeddings import Embedding
from keras.layers import Input
from keras.models import Model
import keras
from keras.preprocessing import sequence
import numpy as np
from sklearn.model_selection import train_test_split


#---------------------------------------------------------#
#******************** PRE PROCESSING *********************#
#---------------------------------------------------------#

sentenceA = []
sentenceB = []
classLabel = []
relatednessScore = []

with open('train.txt') as f:
	for line in f:
		sentence = line.replace('\n', '').replace(',','').replace('\'','').lower().split('\t')
		sentenceA.append(sentence[1])
		sentenceB.append(sentence[2])
		relatednessScore.append(sentence[3])
		classLabel.append(sentence[4])

print(len(sentenceA), len(sentenceB), len(classLabel))

# print(sentenceA)
# exit(0)
words = []

for line in sentenceA:
	for word in line.split():
		words.append(word)

for line in sentenceB:
	for word in line.split():
		words.append(word)

words = sorted(list(set(words)))

# print(words)

print(len(words), "unique words")

# word to index and index to word dictionaries
char_indices = dict((c, i+1) for i, c in enumerate(words))
print("LENGTH CHAR_INDICES: ", len(char_indices))
indices_char = dict((i+1, c) for i, c in enumerate(words))
print("LENGTH INDICES_CHAR: ", len(indices_char))

vectorA = []
vectorB = []

for i in range(len(sentenceA)):
	embeddedVecA = []
	embeddedVecB = []
	for word in sentenceA[i].split():
		embeddedVecA.append(char_indices.get(word))

	for word in sentenceB[i].split():
		embeddedVecB.append(char_indices.get(word))

	embeddedVecA +=  [0]*(40 - len(embeddedVecA))
	embeddedVecB +=  [0]*(40 - len(embeddedVecB))

	vectorA.append(embeddedVecA)
	vectorB.append(embeddedVecB)

# for line in vectorA:
# 	print(line)

encoder = LabelEncoder()
encoder.fit(classLabel)
encoded_Y = encoder.transform(classLabel)
onehot = np_utils.to_categorical(encoded_Y)

'''
CONTRADICTION:  1 0 0
ENTAILMENT: 	0 1 0
NEUTRAL:		0 0 1
'''

#---------------------------------------------------------#
#*********************** LSTM MODEL **********************#
#---------------------------------------------------------#

vectorA = np.array(vectorA)
vectorB = np.array(vectorB)
onehot = np.array(onehot)

print(len(vectorA))
print(len(vectorB))
#X_train, X_test, y_train, y_test = train_test_split([vectorA, vectorB], onehot, test_size=0.30, random_state=7)

# print(X_train[0])
# print(y_train[0])

# for i in vectorB:
# 	if len(i) != 30:
# 		print(i)


print(vectorA.shape)
print(vectorB.shape)

vecAtrain = vectorA[:4000]
vecBtrain = vectorB[:4000]
y_train = onehot[:4000]

vecAtest = vectorA[4000:]
vecBtest = vectorB[4000:]
y_test = onehot[4000:]


#*********** Separate LSTM's for each sentence ******************
# loss: 0.4934 - acc: 0.7853 - val_loss: 0.9514 - val_acc: 0.5680
#****************************************************************

# senA = Input(shape=(40,), dtype='int32', name='senA')
# x = Embedding(output_dim=32, input_dim=2193, input_length=30)(senA)
# senB = Input(shape=(40,), dtype='int32', name='senB')
# y = Embedding(output_dim=32, input_dim=2193, input_length=30)(senB)
# lstm_out1 = LSTM(48)(x)
# # lstm_out1 = LSTM(32)(lstm_out1)
# lstm_out2 = LSTM(48)(y)
# # lstm_out2 = LSTM(32)(lstm_out2)
# x = keras.layers.concatenate([lstm_out1, lstm_out2])
# x = Dense(64, activation='relu')(x)
# x = Dense(32, activation='relu')(x)
# x = Dense(16, activation='relu')(x)
# main_output = Dense(3, activation='softmax', name='main_output')(x)

# model = Model(inputs=[senA, senB], outputs=[main_output])
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy',  metrics=['accuracy'])
# model.fit([vecAtrain, vecBtrain], [y_train], validation_data=([vecAtest, vecBtest], y_test),verbose=1, epochs=25, batch_size=32)

#********************* Shared LSTM Model ************************
# loss: 0.4423 - acc: 0.8097 - val_loss: 1.0831 - val_acc: 0.5560
#****************************************************************

# senA = Input(shape=(40,), dtype='int32', name='senA')
# x = Embedding(output_dim=64, input_dim=2193, input_length=30)(senA)
# senB = Input(shape=(40,), dtype='int32', name='senB')
# y = Embedding(output_dim=64, input_dim=2193, input_length=30)(senB)

# shared_lstm = LSTM(64)

# encoded_a = shared_lstm(x)
# encoded_b = shared_lstm(y)

# merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis=-1)

# main_output = Dense(3, activation='softmax', name='main_output')(merged_vector)

# model = Model(inputs=[senA, senB], outputs=[main_output])
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy',  metrics=['accuracy'])
# model.fit([vecAtrain, vecBtrain], [y_train], validation_data=([vecAtest, vecBtest], y_test),verbose=1, epochs=25, batch_size=32)

#******************** Bi-directional LSTM ***********************
# loss: 0.1873 - acc: 0.9255 - val_loss: 1.9496 - val_acc: 0.5000
#****************************************************************

senA = Input(shape=(40,), dtype='int32', name='senA')
x = Embedding(output_dim=64, input_dim=2193, input_length=30)(senA)
senB = Input(shape=(40,), dtype='int32', name='senB')
y = Embedding(output_dim=64, input_dim=2193, input_length=30)(senB)

shared_lstm = Bidirectional(LSTM(64))

encoded_a = shared_lstm(x)
encoded_b = shared_lstm(y)

merged_vector = keras.layers.concatenate([encoded_a, encoded_b], axis=-1)

main_output = Dense(3, activation='softmax', name='main_output')(merged_vector)

model = Model(inputs=[senA, senB], outputs=[main_output])
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',  metrics=['accuracy'])
model.fit([vecAtrain, vecBtrain], [y_train], validation_data=([vecAtest, vecBtest], y_test),verbose=1, epochs=25, batch_size=32)



# embedding_vecor_length = 32
# model = Sequential()
# model.add(Embedding(2193, embedding_vecor_length, input_length = 30))
# model.add(LSTM(100))
# model.add(Dense(3, activation='softmax'))
# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
