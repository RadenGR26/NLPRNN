import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN, LSTM, GRU, Dense, Bidirectional
from tensorflow.keras.utils import to_categorical

# Load dataset
df = pd.read_csv("data/dataset.csv")

# Label encoding
le = LabelEncoder()
df['label'] = le.fit_transform(df['label'])

# Save label encoder
with open("label_encoder.pkl", "wb") as f:
    pickle.dump(le, f)

# Tokenization
tokenizer = Tokenizer(num_words=5000, oov_token="<OOV>")
tokenizer.fit_on_texts(df['text'])

# Save tokenizer
with open("tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

# Preprocessing
sequences = tokenizer.texts_to_sequences(df['text'])
X = pad_sequences(sequences, padding='post', maxlen=100)
y = to_categorical(df['label'])

# Model builder
def build_model(cell, name):
    model = Sequential([
        Embedding(input_dim=5000, output_dim=64, input_length=100),
        cell,
        Dense(2, activation='softmax')
    ])
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X, y, epochs=10, batch_size=8, verbose=1)
    model.save(f"models/{name}.h5")

# Create models directory
os.makedirs("models", exist_ok=True)

# Build each model
build_model(SimpleRNN(64), "rnn_model")
build_model(LSTM(64), "lstm_model")
build_model(GRU(64), "gru_model")
build_model(Bidirectional(SimpleRNN(64)), "birnn_model")
