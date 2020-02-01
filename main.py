from os import getenv as env
from dotenv import load_dotenv

from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.optimizers import Adam

from model.embedding import build_embedding
from model.attention import build_attention, build_bi_lstm, build_classifier
from utils.gpu_limit_session import GpuLimitSession

from preprocessing.dataset_preprocessing import get_dataset


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


if __name__ == "__main__":
    load_dotenv()

    # load dataset for training
    raw_data_dir = env("RAW_DATA_IMDB_DIR")
    pickle_train_dir = env("PICKLE_TRAIN_DIR")
    pickle_test_dir = env("PICKLE_TEST_DIR")
    train_seqs, test_seqs = get_dataset(raw_data_dir, pickle_train_dir, pickle_test_dir)

    # with GpuLimitSession(gpu_factor=0.7, disable_eager=True) as gpu:
    #     main()  # run the main function
