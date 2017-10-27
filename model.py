#import statements
from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
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

	embeddedVecA +=  [0]*(30 - len(embeddedVecA))
	embeddedVecB +=  [0]*(30 - len(embeddedVecB))

	vectorA.append(embeddedVecA)
	vectorB.append(embeddedVecB)

for line in vectorA:
	print(line)


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
onehot = np.array(onehot)

X_train, X_test, y_train, y_test = train_test_split(vectorA, onehot, test_size=0.30, random_state=7)

embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(2193, embedding_vecor_length, input_length = 30))
model.add(LSTM(100))
model.add(Dense(3, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3, batch_size=64)
