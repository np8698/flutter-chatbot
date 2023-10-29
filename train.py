import random
from tensorflow.keras.optimizers import legacy
from keras.layers import Dense, Dropout
from keras.models import load_model
from keras.models import Sequential
import numpy as np
import pickle
import json
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
nltk.download('omw-1.4')
nltk.download("punkt")
nltk.download("wordnet")


# init file
words = []
classes = []
documents = []
ignore_words = ["?", "!"]
data_file = open("intents.json").read()
intents = json.loads(data_file)

# words
for intent in intents["intents"]:
    for pattern in intent["patterns"]:

        # take each word and tokenize it
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        # adding documents
        documents.append((w, intent["tag"]))

        # adding classes to our class list
        if intent["tag"] not in classes:
            classes.append(intent["tag"])

# lemmatizer
words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

print(len(documents), "documents")

print(len(classes), "classes", classes)

print(len(words), "unique lemmatized words", words)


pickle.dump(words, open("words.pkl", "wb"))
pickle.dump(classes, open("classes.pkl", "wb"))

# initializing training data
training = [[] for _ in range(len(documents))]

output_empty = [0] * len(classes)
for doc in documents:
  # initializing bag of words
  bag = []
  # list of tokenized words for the pattern
  pattern_words = doc[0]
  # lemmatize each word - create base word, in attempt to represent related words
  pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
  # create our bag of words array with 1, if word match found in current pattern
  for w in words:
    bag.append(1) if w in pattern_words else bag.append(0)

  # output is a '0' for each tag and '1' for current tag (for each pattern)
  output_row = list(output_empty)
  output_row[classes.index(doc[1])] = 1

  # add the bag of words and output label to the training data
  training[documents.index(doc)].append(bag)
  training[documents.index(doc)].append(output_row)

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training, dtype=object)

# create train and test lists. X - patterns, Y - intents
train_x = list(np.array(training).T[0])
train_y = list(np.array(training).T[1])
print("Training data created")

# actual training
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation="softmax"))
model.summary()

# Compile model. Stochastic gradient descent with Nesterov accelerated gradient gives good results for this model
sgd = legacy.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

#Optional snippet - I have commented it as I did not use it.

# from keras import callbacks 
# earlystopping = callbacks.EarlyStopping(monitor ="loss", mode ="min", patience = 5, restore_best_weights = True)
# callbacks =[earlystopping]

# fitting and saving the model
train_x_array = np.zeros((len(train_x), len(words)))
for i in range(len(train_x)):
  train_x_array[i] = train_x[i][0]

train_y_array = np.zeros((len(train_y), len(words)))
for i in range(len(train_y)):
  train_y_array[i] = train_y[i][0]

hist = model.fit(train_x_array, train_y_array, epochs=200, batch_size=5, verbose=1)

model.save("chatbot_model.h5", hist)
print("model created")