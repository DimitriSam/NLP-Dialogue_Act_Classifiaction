import itertools
import pickle
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical


def generate_embeddings(data, metadata, verbose=False):
    word_to_index = metadata['word_to_index']
    max_utterance_len = metadata['max_utterance_len']

    label_to_index = metadata['label_to_index']
    num_labels = metadata['num_labels']

    utterances = data['utterances']
    labels = data['labels']

    tmp_utterance_embeddings = []
    tmp_label_embeddings = []

    # Convert each word and label into its numerical representation
    for i in range(len(utterances)):

        tmp_utt = []
        for word in utterances[i]:
            tmp_utt.append(word_to_index[word])

        tmp_utterance_embeddings.append(tmp_utt)
        tmp_label_embeddings.append(label_to_index[labels[i]])

    # For Keras LSTM must pad the sequences to same length and return a numpy array
    utterance_embeddings = pad_sequences(tmp_utterance_embeddings, maxlen=max_utterance_len, padding='post', value=0.0)

    # Convert labels to one hot vectors
    label_embeddings = to_categorical(np.asarray(tmp_label_embeddings), num_classes=num_labels)

    if verbose:
        print("------------------------------------")
        print("Created utterance/label embeddings, and padded utterances...")
        print("Number of utterances: ", utterance_embeddings.shape[0])

    return utterance_embeddings, label_embeddings


def generate_probabilistic_embeddings(data, frequency_data, metadata, verbose=False):
    freq_words = frequency_data['freq_words']
    probability_matrix = frequency_data['probability_matrix']

    word_to_index = metadata['word_to_index']
    max_utterance_len = metadata['max_utterance_len']

    label_to_index = metadata['label_to_index']
    num_labels = metadata['num_labels']

    utterances = data['utterances']
    labels = data['labels']

    tmp_label_embeddings = []

    # Convert each word and label into its numerical representation
    utterance_embeddings = np.zeros((len(utterances), max_utterance_len, num_labels))
    for i in range(len(utterances)):

        for j in range(len(utterances[i])):
            word = utterances[i][j]
            if word in freq_words:
                utterance_embeddings[i][j] = probability_matrix[word_to_index[word]]

        tmp_label_embeddings.append(label_to_index[labels[i]])

    # Convert labels to one hot vectors
    label_embeddings = to_categorical(np.asarray(tmp_label_embeddings), num_classes=num_labels)

    if verbose:
        print("------------------------------------")
        print("Created utterance/label embeddings, and padded utterances...")
        print("Number of utterances: ", utterance_embeddings.shape[0])

    return utterance_embeddings, label_embeddings





