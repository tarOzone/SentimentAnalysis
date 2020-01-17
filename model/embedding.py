from tensorflow.keras.layers import Embedding


def build_embedding(inputs, num_words, maxlen):
    embedding_layer = Embedding(num_words, maxlen)(inputs)
    return embedding_layer
