import re
import json
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences


class SentencePreprocessor:

    def __init__(self, maxlen, num_words):
        self.maxlen = maxlen
        self.num_words = num_words
        self.word_index = self._read_imdb_word_index()
        self.word_index = {k: (v + 3) for k, v in self.word_index.items()}
        self.word_index["<PAD>"] = 0
        self.word_index["<START>"] = 1
        self.word_index["<UNK>"] = 2
        self.word_index["<UNUSED>"] = 3
        self.rev_word_index = {v: k for k, v in self.word_index.items()}

    def encode_sentence(self, sentence):
        sentence = self._tokenize(sentence)
        sentence = sentence.split(" ")
        sentence = ["<START>"] + sentence if sentence[0] != "<START>" else sentence
        encoded_sentence = [self.word_index.get(s, 2) for s in sentence]
        return self._pad_sentence(encoded_sentence)[0]

    def pad_dataset(self, data):
        return pad_sequences(
            data,
            value=self.word_index["<PAD>"],
            padding="post",
            maxlen=self.maxlen
        )

    def _tokenize(self, sentence):
        sentence = sentence.lower()
        return re.sub("[^a-zA-Z' ]+", "", sentence)

    def _pad_sentence(self, sentence):
        return self.pad_dataset(np.array([sentence]))

    def _read_imdb_word_index(self):
        with open("../dataset/imdb_word_index.json", "r") as f:
            content = f.read()
        return json.loads(content)

    def decode_sentence(self, encoded_sentence, capitalize=True):
        encoded_sentence = [s for s in encoded_sentence if s not in [0, 1]]
        sentence = " ".join([self.rev_word_index.get(i, "?") for i in encoded_sentence])
        return sentence.capitalize() if capitalize else sentence
