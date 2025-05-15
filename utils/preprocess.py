# Third-party imports
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences
from sklearn.preprocessing import LabelEncoder
from transformers import BertTokenizer


class GloVeEmbeddings:
    """
    A class for loading and creating GloVe embeddings.
    """

    def __init__(self, glove_file_path, embedding_dim):
        """
        Initialize the GloVeEmbeddings.

        Args:
            glove_file_path (str): The path to the GloVe embeddings file.
            embedding_dim (int): The dimension of the embeddings.
        """
        self.glove_file_path = glove_file_path
        self.embedding_dim = embedding_dim
        self.embeddings_index = self._load_glove_embeddings()

    def _load_glove_embeddings(self):
        """
        Load the GloVe embeddings from the file.

        Returns:
            dict: A dictionary mapping words to their embeddings.
        """
        embeddings_index = {}
        try:
            with open(self.glove_file_path, 'r', encoding='utf8') as file:
                for line in file:
                    values = line.split()
                    word = values[0]
                    coefs = np.asarray(values[1:], dtype='float32')
                    embeddings_index[word] = coefs
        except FileNotFoundError:
            raise Exception(f"GloVe file not found at path: {self.glove_file_path}")

        return embeddings_index

    def create_embedding_matrix(self, word_index):
        """
        Create an embedding matrix based on the word index.

        Args:
            word_index (dict): A dictionary mapping words to their indices.

        Returns:
            np.ndarray: The embedding matrix.
        """
        embedding_matrix = np.zeros((len(word_index) + 1, self.embedding_dim))
        for word, i in word_index.items():
            embedding_vector = self.embeddings_index.get(word)
            if embedding_vector is not None:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix


class DataPreprocessor:
    """
    A class for preprocessing textual data.
    """

    def __init__(self):
        self.tokenizer = Tokenizer()
        self.label_encoder = LabelEncoder()
        self.max_seq_length = 0
        self.vocab_size = 0
        self.num_classes = 0

    def preprocess(self, train_data, test_data):
        """
        Preprocess the training and testing data.

        Args:
            train_data (list): The training data.
            test_data (list): The testing data.

        Returns:
            tuple: A tuple containing the preprocessed data:
                - train_padded (np.ndarray): The padded training sequences.
                - train_labels_encoded (np.ndarray): The encoded training labels.
                - test_padded (np.ndarray): The padded testing sequences.
                - test_labels_encoded (np.ndarray): The encoded testing labels.
        """
        train_sentences, train_labels = self._extract_sentences_and_labels(train_data)
        test_sentences, test_labels = self._extract_sentences_and_labels(test_data)

        self._fit_tokenizer_and_encoder(train_sentences, test_sentences, train_labels, test_labels)
        train_sequences = self.tokenizer.texts_to_sequences(train_sentences)
        test_sequences = self.tokenizer.texts_to_sequences(test_sentences)

        self._update_max_seq_length(train_sequences, test_sequences)
        train_padded = pad_sequences(train_sequences, maxlen=self.max_seq_length, padding='post')
        test_padded = pad_sequences(test_sequences, maxlen=self.max_seq_length, padding='post')

        train_labels_encoded = to_categorical(self.label_encoder.transform(train_labels))
        test_labels_encoded = to_categorical(self.label_encoder.transform(test_labels))

        return train_padded, train_labels_encoded, test_padded, test_labels_encoded

    def preprocess_single_sentence(self, sentence):
        """
        Preprocess a single sentence.

        Args:
            sentence (str): The input sentence.

        Returns:
            np.ndarray: The padded sequence.
        """
        sequence = self.tokenizer.texts_to_sequences([sentence])
        padded = pad_sequences(sequence, maxlen=self.max_seq_length, padding='post')
        return padded

    def _fit_tokenizer_and_encoder(self, train_sentences, test_sentences, train_labels, test_labels):
        """
        Fit the tokenizer and label encoder on the training and testing data.

        Args:
            train_sentences (list): The training sentences.
            test_sentences (list): The testing sentences.
            train_labels (list): The training labels.
            test_labels (list): The testing labels.
        """
        self.tokenizer.fit_on_texts(train_sentences + test_sentences)
        self.vocab_size = len(self.tokenizer.word_index) + 1
        self.label_encoder.fit(train_labels + test_labels)
        self.num_classes = len(self.label_encoder.classes_)

    def _update_max_seq_length(self, train_sequences, test_sequences):
        """
        Update the maximum sequence length based on the training and testing sequences.

        Args:
            train_sequences (list): The training sequences.
            test_sequences (list): The testing sequences.
        """
        self.max_seq_length = max(max(len(seq) for seq in train_sequences),
                                  max(len(seq) for seq in test_sequences))

    @staticmethod
    def _extract_sentences_and_labels(data):
        """
        Extract sentences and labels from the data.

        Args:
            data (list): The input data.

        Returns:
            tuple: A tuple containing the sentences and labels.
        """
        sentences = [item['sentence'] for item in data]
        labels = [item['relation'] for item in data]
        return sentences, labels

    def get_vocab_size(self):
        """
        Get the vocabulary size.

        Returns:
            int: The vocabulary size.
        """
        return self.vocab_size

    def get_num_classes(self):
        """
        Get the number of classes.

        Returns:
            int: The number of classes.
        """
        return self.num_classes

    def get_max_seq_length(self):
        """
        Get the maximum sequence length.

        Returns:
            int: The maximum sequence length.
        """
        return self.max_seq_length

    def get_class_names(self):
        """
        Get the class names.

        Returns:
            np.ndarray: The class names.
        """
        return self.label_encoder.classes_


class BertTinyDataPreprocessor:
    """
    A class for preprocessing data using BERT Tiny.
    """

    def __init__(self, bert_model_name='google/bert_uncased_L-2_H-128_A-2', max_seq_length=128):
        """
        Initialize the BertTinyDataPreprocessor.

        Args:
            bert_model_name (str): The name of the BERT model.
            max_seq_length (int): The maximum sequence length.
        """
        self.tokenizer = BertTokenizer.from_pretrained(bert_model_name)
        self.max_seq_length = max_seq_length
        self.label_encoder = LabelEncoder()
        self.num_classes = 0

    def preprocess(self, train_data, test_data):
        """
        Preprocess the training and testing data.

        Args:
            train_data (list): The training data.
            test_data (list): The testing data.

        Returns:
            tuple: A tuple containing the preprocessed data:
                - train_encodings (tuple): The encoded training data.
                - train_labels_encoded (np.ndarray): The encoded training labels.
                - test_encodings (tuple): The encoded testing data.
                - test_labels_encoded (np.ndarray): The encoded testing labels.
        """
        train_sentences, train_labels = self._extract_sentences_and_labels(train_data)
        test_sentences, test_labels = self._extract_sentences_and_labels(test_data)

        train_encodings = self._tokenize_sentences(train_sentences)
        test_encodings = self._tokenize_sentences(test_sentences)
        self._fit_label_encoder(train_labels, test_labels)

        train_labels_encoded = to_categorical(self.label_encoder.transform(train_labels))
        test_labels_encoded = to_categorical(self.label_encoder.transform(test_labels))

        return (train_encodings['input_ids'], train_encodings['attention_mask']), train_labels_encoded, (
            test_encodings['input_ids'], test_encodings['attention_mask']), test_labels_encoded

    def preprocess_single_sentence(self, sentence):
        """
        Preprocess a single sentence.

        Args:
            sentence (str): The input sentence.

        Returns:
            tuple: A tuple containing the encoded input IDs and attention mask.
        """
        encoding = self.tokenizer(sentence, truncation=True, padding='max_length',
                                  max_length=self.max_seq_length, return_tensors='tf')
        return encoding['input_ids'], encoding['attention_mask']

    def _tokenize_sentences(self, sentences):
        """
        Tokenize the sentences using the BERT tokenizer.

        Args:
            sentences (list): The list of sentences.

        Returns:
            dict: A dictionary containing the encoded input IDs and attention masks.
        """
        return self.tokenizer(sentences, truncation=True, padding='max_length',
                              max_length=self.max_seq_length, return_tensors='tf')

    def _fit_label_encoder(self, train_labels, test_labels):
        """
        Fit the label encoder on the training and testing labels.

        Args:
            train_labels (list): The training labels.
            test_labels (list): The testing labels.
        """
        self.label_encoder.fit(train_labels + test_labels)
        self.num_classes = len(self.label_encoder.classes_)

    @staticmethod
    def _extract_sentences_and_labels(data):
        """
        Extract sentences and labels from the data.

        Args:
            data (list): The input data.

        Returns:
            tuple: A tuple containing the sentences and labels.
        """
        sentences = [item['sentence'] for item in data]
        labels = [item['relation'] for item in data]
        return sentences, labels

    def get_num_classes(self):
        """
        Get the number of classes.

        Returns:
            int: The number of classes.
        """
        return self.num_classes

    def get_class_names(self):
        """
        Get the class names.

        Returns:
            np.ndarray: The class names.
        """
        return self.label_encoder.classes_

    def get_max_seq_length(self):
        """
        Get the maximum sequence length.

        Returns:
            int: The maximum sequence length.
        """
        return self.max_seq_length
    