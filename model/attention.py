from tensorflow.keras import backend as K
from tensorflow.keras.layers import Multiply, Lambda
from tensorflow.keras.layers import Bidirectional, LSTM, GRU
from tensorflow.keras.layers import Dense, Flatten, Activation, RepeatVector, Permute


def build_bi_lstm(embedded, lstm_units=64, rnn_model='lstm', dropout=0.2, recurrent_dropout=0.2):
    if rnn_model.lower() == 'lstm':
        rnn = LSTM
    elif rnn_model.lower() == 'gru':
        rnn = GRU
    else:
        raise AttributeError('rnn_model must be either "lstm" or "gru".')
    rnn = rnn(lstm_units, return_sequences=True, dropout=dropout, recurrent_dropout=recurrent_dropout)
    bi_lstm = Bidirectional(rnn)(embedded)
    return bi_lstm


def build_attention(bi_lstm, lstm_units, activation='tanh'):
    attention = bi_lstm
    # you can model layers here

    attention = Dense(1, activation=activation)(attention)
    attention = Flatten()(attention)
    attention = Activation('softmax')(attention)
    attention = RepeatVector(lstm_units * 2)(attention)
    attention = Permute([2, 1])(attention)
    return attention


def build_classifier(bi_lstm, attention, lstm_units):
    classifier = Multiply()([bi_lstm, attention])
    classifier = Lambda(lambda x_in: K.sum(x_in, axis=-2), output_shape=(lstm_units * 2,))(classifier)
    # you can model layers here

    classifier = Dense(1, activation='sigmoid')(classifier)
    return classifier
