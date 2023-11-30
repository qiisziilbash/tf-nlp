# %% Params
vocab_size = 10000
embedding_dim = 16
max_length = 100
trunc_type = "post"
padding_type = "post"
oov_tok = "<OOV>"
train_ratio = 0.8

# %%
import json
import pathlib

import requests

datafile_name = "sarcasm.json"

if not pathlib.Path(datafile_name).exists:
    data = requests.get("https://storage.googleapis.com/learning-datasets/sarcasm.json")
    with open(datafile_name, "w") as file:
        file.write(str(json.dumps(data.json())))

with open("sarcasm.json", "r") as f:
    datastore = json.load(f)


sentences, labels = [], []
for item in datastore:
    sentences.append(item["headline"])
    labels.append(item["is_sarcastic"])


training_size = int(train_ratio * len(sentences))
training_sentences = sentences[0:training_size]
testing_sentences = sentences[training_size:]

training_labels = labels[0:training_size]
testing_labels = labels[training_size:]

# %%
from tensorflow import keras

tokenizer = keras.preprocessing.text.Tokenizer(num_words=vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(training_sentences)


training_sequences = tokenizer.texts_to_sequences(training_sentences)
training_padded = keras.preprocessing.sequence.pad_sequences(
    training_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type
)

testing_sequences = tokenizer.texts_to_sequences(testing_sentences)
testing_padded = keras.preprocessing.sequence.pad_sequences(
    testing_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type
)

# %%
import numpy as np

training_padded = np.array(training_padded)
training_labels = np.array(training_labels)
testing_padded = np.array(testing_padded)
testing_labels = np.array(testing_labels)

model = keras.Sequential(
    [
        keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(
            24,
            activation="relu",
        ),
        keras.layers.Dense(1, activation="sigmoid"),
    ]
)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()

# %%
num_epochs = 30
history = model.fit(
    training_padded,
    training_labels,
    epochs=num_epochs,
    validation_data=(testing_padded, testing_labels),
    verbose=2,
)

import matplotlib.pyplot as plt


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history["val_" + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, "val_" + string])
    plt.show()


plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

# %%
sentence = [
    "granny starting to fear spiders in the garden might be real",
    "game of thrones season finale showing this sunday night",
]
sequences = tokenizer.texts_to_sequences(sentence)
padded = keras.preprocessing.sequence.pad_sequences(
    sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type
)
print(model.predict(padded))

# %%
