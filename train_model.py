import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
import pickle

def main():
    # Load cleaned data
    df = pd.read_csv("cleaned_comments.csv")

    # Prepare data
    max_words = 5000
    max_len = 50
    tokenizer = Tokenizer(num_words=max_words, oov_token="<OOV>")
    tokenizer.fit_on_texts(df["cleaned_text"])

    X = tokenizer.texts_to_sequences(df["cleaned_text"])
    X = pad_sequences(X, maxlen=max_len, padding="post")
    y = to_categorical(df["sentiment"] + 1, num_classes=3)

    # Build model
    model = Sequential([
        Embedding(input_dim=max_words, output_dim=128, input_length=max_len),
        LSTM(64, return_sequences=True),
        Dropout(0.3),
        LSTM(32),
        Dense(3, activation="softmax")
    ])
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.build(input_shape=(None, max_len))
    model.summary()
    model.fit(X, y, epochs=20, batch_size=32, validation_split=0.2)

    # Save model and tokenizer
    model.save("sentiment_model.h5")
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)

    print("✅ Model and tokenizer saved.")

if __name__ == "__main__":
    main()
