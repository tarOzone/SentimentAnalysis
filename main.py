from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam

from model.attention import build_attention, build_bi_lstm, build_classifier
from utils.gpu_limiter import Session


def main(*args, **kwargs):

    # model parameters
    lstm_units = 256
    learning_rate = 3e-4
    inputs = Input(shape=(1000, 2048))

    bi_lstm = build_bi_lstm(inputs, lstm_units)  # build the bidirectional LSTM model
    attention = build_attention(bi_lstm, lstm_units)  # build the attention model
    classifier = build_classifier(bi_lstm, attention, lstm_units)   # build the classifier model

    # build the complete model
    model = Model(inputs, classifier)

    # init optimizer and loss function
    opt = Adam(lr=learning_rate)
    loss = binary_crossentropy
    model.compile(loss=loss, optimizer=opt, metrics=['accuracy'])
    model.summary()


if __name__ == "__main__":
    sess = Session(gpu_factor=0.7)  # init Session for GPU limiter
    main()  # run the main function
    del sess    # remember to close the session


