from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Embedding
from tensorflow.keras.optimizers import Adam
from preprocessing.dataset_preprocessing import *
from model.attention import build_attention, build_bi_lstm, build_classifier

import pickle
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences


def build_model(max_words, max_length_sentence, learning_rate=1e-3, summary=True):
    # init inputs
    inputs = Input(shape=(max_length_sentence,))
    embedding_layer = Embedding(max_words, max_length_sentence)(inputs)

    # building the model
    lstm_units = 200
    bi_lstm = build_bi_lstm(embedding_layer, lstm_units)  # build the bidirectional LSTM model
    attention = build_attention(bi_lstm, lstm_units)  # build the attention model
    classifier = build_classifier(bi_lstm, attention, lstm_units)  # build the classifier model

    # compile model with optimizer and loss function
    model = Model(inputs, classifier)
    model.compile(loss='binary_crossentropy', optimizer=Adam(lr=learning_rate), metrics=['accuracy'])
    if summary:
        model.summary()
    return model


def load_dataset(df, target_columns):
    targets = df[[target_columns]]
    pd.DataFrame.pop(df, target_columns)
    features = df
    return features.to_numpy(), targets.to_numpy()


# def preprocess(texts, seq_len, num_words=8000):
#     x_list = list(texts[:, 0])
#     tokenizer = Tokenizer(num_words=num_words, oov_token='<UNK>')
#     tokenizer.fit_on_texts(x_list)
#     seqs = tokenizer.texts_to_sequences(x_list)
#     seqs = pad_sequences(seqs, maxlen=seq_len, padding="post")
#     return seqs
# def get_dataset(df, seq_len, shuffle_buffer_data, batch_size, max_words):
#     X, y = load_dataset(df, 'sent')
#     X_prep = preprocess(X, seq_len=seq_len, num_words=max_words)
#     tensor_data = tf.data.Dataset.from_tensor_slices((X_prep, y))
#     return tensor_data.shuffle(shuffle_buffer_data).batch(batch_size)

def to_tensor_dataset(X, y, tokenizer, seq_len, shuffle_buffer_data, batch_size):
    x_list = list(X[:, 0])
    X_prep = tokenizer.texts_to_sequences(x_list)
    X_prep = pad_sequences(X_prep, maxlen=seq_len, padding="post")
    dataset = tf.data.Dataset.from_tensor_slices((X_prep, y))
    return dataset.shuffle(shuffle_buffer_data).batch(batch_size)


def get_train_test_datatset(train_df, test_df, seq_len=200, shuffle_buffer_data=100, batch_size=4, max_words=8000):
    X_train, y_train = load_dataset(train_df, 'sent')
    X_test, y_test = load_dataset(test_df, 'sent')

    tokenizer = Tokenizer(num_words=max_words, oov_token='<UNK>')
    tokenizer.fit_on_texts(list(X_train[:, 0]))

    train_dataset = to_tensor_dataset(X_train, y_train, tokenizer, seq_len, shuffle_buffer_data, batch_size)
    test_dataset = to_tensor_dataset(X_test, y_test, tokenizer, seq_len, shuffle_buffer_data, batch_size)
    return train_dataset, test_dataset


if __name__ == "__main__":
    with open("data/imdb_train.pickle", "rb") as pkl:
        df_train = pickle.load(pkl)
    with open("data/imdb_test.pickle", "rb") as pkl:
        df_test = pickle.load(pkl)

    seq_len = 200
    batch_size = 4
    max_words = 8000
    shuffle_buffer_data = 100

    train_dataset, test_dataset = get_train_test_datatset(df_train, df_test)

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(max_words, seq_len),
        tf.keras.layers.GlobalAveragePooling1D(),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.summary()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    try:
        model.fit(train_dataset, epochs=3)
    finally:
        model.save("saved_model")
        model.evaluate(test_dataset)
