from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import pandas as pd
import pickle
import os


def get_dataframe(start_path):
    df = pd.DataFrame(columns=['text', 'sent'])
    text = []
    sent = []
    for p in ['pos', 'neg']:
        path = os.path.join(start_path, p)
        files = [f for f in os.listdir(path) if os.path.isfile(os.path.join(path, f))]
        for f in files:
            print("*", f)
            with open(os.path.join(path, f), "r", encoding='utf-8') as FILE:
                s = FILE.read()
                text.append(s.replace("\n", " ").replace("\r", " "))
                sent.append(1 if p == 'pos' else 0)
    df['text'] = text
    df['sent'] = sent
    # This line shuffles the data so you don't end up with contiguous
    # blocks of positive and negative reviews
    return df.sample(frac=1).reset_index(drop=True)


def preprocess(train_df, test_df, NUM_WORDS=8000, SEQ_LEN=100):
    # create tokenizer for our data
    tokenizer = Tokenizer(num_words=NUM_WORDS, oov_token='<UNK>')
    tokenizer.fit_on_texts(train_df['text'])
    # convert text data to numerical indexes
    train_seqs = tokenizer.texts_to_sequences(train_df['text'])
    test_seqs = tokenizer.texts_to_sequences(test_df['text'])
    # pad data up to SEQ_LEN (note that we truncate if there are more than SEQ_LEN tokens)
    train_seqs = pad_sequences(train_seqs, maxlen=SEQ_LEN, padding="post")
    test_seqs = pad_sequences(test_seqs, maxlen=SEQ_LEN, padding="post")
    return train_seqs, test_seqs


def get_dataset_from_raw(raw_data_dir, pickle_train_dir, pickle_test_dir, SEQ_LEN):
    # collect raw Imdb data into dataframe
    train_df = get_dataframe(os.path.join(raw_data_dir, "train"))
    test_df = get_dataframe(os.path.join(raw_data_dir, "test"))

    # then pass those dataframe into this function to make it
    train_seqs, test_seqs = preprocess(train_df, test_df, SEQ_LEN=SEQ_LEN)

    # to save your time, pickle both
    with open(pickle_train_dir, "wb") as f:
        pickle.dump(train_seqs, f)
    with open(pickle_test_dir, "wb") as f:
        pickle.dump(test_seqs, f)
    return train_seqs, test_seqs


def get_dataset(raw_data_dir, pickle_train_dir, pickle_test_dir, SEQ_LEN=128):
    if os.path.exists(pickle_train_dir) and os.path.exists(pickle_test_dir):
        with open(pickle_train_dir, "rb") as pck_f:
            train_seqs = pickle.load(pck_f)
        with open(pickle_test_dir, "rb") as pck_f:
            test_seqs = pickle.load(pck_f)
        print("[INFO] get data from pickle")
        return train_seqs, test_seqs
    print("[INFO] get data from Imdb")
    return get_dataset_from_raw(raw_data_dir, pickle_train_dir, pickle_test_dir, SEQ_LEN=SEQ_LEN)
