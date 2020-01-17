from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam

from model.attention import build_attention, build_bi_lstm, build_classifier
from utils.gpu_limiter import Session


def main(*args, **kwargs):
    # model parameters
    max_length_sentence = 1000
    embedded_features = 2048
    lstm_units = 256
    learning_rate = 3e-4
    inputs = Input(shape=(max_length_sentence, embedded_features))

    # building the model
    bi_lstm = build_bi_lstm(inputs, lstm_units)  # build the bidirectional LSTM model
    attention = build_attention(bi_lstm, lstm_units)  # build the attention model
    classifier = build_classifier(bi_lstm, attention, lstm_units)   # build the classifier model
    # build the complete model
    model = Model(inputs, classifier)

    # compile model with optimizer and loss function
    opt = Adam(lr=learning_rate)
    loss = binary_crossentropy
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    model.summary()


def main2():
    import numpy as np
    from tensorflow.keras.datasets import imdb

    num_words = 20000
    (train_x, train_y), (test_x, test_y) = imdb.load_data(num_words=num_words)

    print(max([len(x) for x in train_x]))
    print(max([len(x) for x in test_x]))




if __name__ == "__main__":
    sess = Session(gpu_factor=0.7, disable_eager=True)  # init Session for GPU limiter
    main2()  # run the main function
    del sess    # remember to close the session


