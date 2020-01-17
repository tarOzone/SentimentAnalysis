# from tensorflow.keras import Model
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input
# from tensorflow.keras.optimizers import Adam
#
# from model.attention import build_attention, build_bi_lstm, build_classifier
#
#
# class ModelBuilder(Model):
#
#     def __init__(self, max_len, embedded_features, learning_rate):
#         # model parameters
#         self.lstm_units = 256
#         self.inputs = Input(shape=(max_len, embedded_features))
#
#         # building the model
#         bi_lstm = build_bi_lstm(self.inputs, self.lstm_units)  # build the bidirectional LSTM model
#         attention = build_attention(bi_lstm, self.lstm_units)  # build the attention model
#         classifier = build_classifier(bi_lstm, attention, self.lstm_units)  # build the classifier model
#
#         # build the complete model
#         model = Model(self.inputs, classifier)
#
#         # compile model with optimizer and loss function
#         opt = Adam(lr=learning_rate)
#         model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
#         model.summary()
