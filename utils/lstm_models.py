# Standard library imports
import numpy as np

# Third-party library imports
from keras.models import Sequential, Model, load_model
from keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional, GlobalMaxPooling1D, GlobalAveragePooling1D, Input, Attention
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from transformers import TFBertModel

# Global function definitions
def get_callbacks(patience=10, model_name="model.h5"):
    """
    Generate a list of callbacks for training the model.

    :param patience: Number of epochs with no improvement after which training will be stopped.
    :param model_name: Name of the file where the model will be saved.
    :return: List of callbacks.
    """
    early_stopping = EarlyStopping(monitor='accuracy', patience=patience, verbose=1, mode='max')
    model_checkpoint = ModelCheckpoint(f"checkpoints/{model_name}", save_best_only=True, monitor='accuracy', mode='max')
    reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=5, min_lr=1e-6)
    return [early_stopping, model_checkpoint, reduce_lr]


# Class definitions
class BaseModel:
    """
    A base class for different LSTM models.
    """
    def __init__(self, vocab_size, max_seq_length, num_classes):
        self.model = None
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.num_classes = num_classes

    def summary(self):
        """
        Print the summary of the model.
        """
        if self.model:
            return self.model.summary()
        raise NotImplementedError("Model not defined.")

    def train(self, train_padded, train_labels, validation_split=0.1, epochs=10, batch_size=64, callbacks=None):
        """
        Train the model with the given data.

        :param train_padded: Padded training data.
        :param train_labels: Labels for the training data.
        :param validation_split: Fraction of the training data to be used as validation data.
        :param epochs: Number of epochs to train the model.
        :param batch_size: Number of samples per gradient update.
        :param callbacks: List of callbacks to apply during training.
        :return: A history object.
        """
        if not self.model:
            raise NotImplementedError("Model not defined.")

        if callbacks is None:
            model_name = type(self).__name__ + "_model.h5"
            callbacks = get_callbacks(model_name=model_name)
        return self.model.fit(
            train_padded, train_labels, epochs=epochs, batch_size=batch_size, 
            validation_split=validation_split, callbacks=callbacks
        )

    def evaluate(self, test_padded, test_labels):
        """
        Evaluate the model on the test set.

        :param test_padded: Padded test data.
        :param test_labels: Labels for the test data.
        :return: Loss value & metrics values for the model in test mode.
        """
        if not self.model:
            raise NotImplementedError("Model not defined.")
        return self.model.evaluate(test_padded, test_labels)

class BaseLSTM(BaseModel):
    """
    A class representing a basic LSTM model.
    """
    def __init__(self, vocab_size, max_seq_length, num_classes):
        super().__init__(vocab_size, max_seq_length, num_classes)
        self.model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=256, input_length=max_seq_length),
            LSTM(128, return_sequences=True),
            LSTM(128),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        self.model.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

class BiLSTM(BaseModel):
    """
    A class representing a Bidirectional LSTM model.
    """
    def __init__(self, vocab_size, max_seq_length, num_classes):
        super().__init__(vocab_size, max_seq_length, num_classes)
        self.model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=256, input_length=max_seq_length),
            Bidirectional(LSTM(128, return_sequences=True)),
            Bidirectional(LSTM(128)),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        self.model.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

class RegularizedBiLSTM(BaseModel):
    """
    A class representing a Regularized Bidirectional LSTM model with dropout.
    """
    def __init__(self, vocab_size, max_seq_length, num_classes):
        super().__init__(vocab_size, max_seq_length, num_classes)
        self.model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=256, input_length=max_seq_length),
            Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),
            Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)),
            Dense(64, activation='relu'),
            Dropout(0.2),
            Dense(num_classes, activation='softmax')
        ])
        self.model.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

class AvgPoolingBiLSTM(BaseModel):
    """
    A class representing a Bidirectional LSTM model with Average pooling.
    """
    def __init__(self, vocab_size, max_seq_length, num_classes):
        super().__init__(vocab_size, max_seq_length, num_classes)
        base_model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=256, input_length=max_seq_length),
            Bidirectional(LSTM(128, return_sequences=True)),
            Bidirectional(LSTM(128, return_sequences=True))
        ])

        avg_pool = GlobalAveragePooling1D()(base_model.output)
        dense = Dense(64, activation='relu')(avg_pool)
        dropout = Dropout(0.2)(dense)
        output = Dense(num_classes, activation='softmax')(dropout)

        self.model = Model(inputs=base_model.input, outputs=output)
        self.model.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

class MaxPoolingBiLSTM(BaseModel):
    """
    A class representing a Bidirectional LSTM model with Max Pooling.
    """
    def __init__(self, vocab_size, max_seq_length, num_classes):
        super().__init__(vocab_size, max_seq_length, num_classes)
        base_model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=256, input_length=max_seq_length),
            Bidirectional(LSTM(128, return_sequences=True)),
            Bidirectional(LSTM(128, return_sequences=True))
        ])

        max_pool = GlobalMaxPooling1D()(base_model.output)
        dense = Dense(64, activation='relu')(max_pool)
        dropout = Dropout(0.2)(dense)
        output = Dense(num_classes, activation='softmax')(dropout)

        self.model = Model(inputs=base_model.input, outputs=output)
        self.model.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])


class AttentionBiLSTM(BaseModel):
    """
    A class representing a Bidirectional LSTM model with an Seq2Seq Attention layer.
    """
    def __init__(self, vocab_size, max_seq_length, num_classes):
        super().__init__(vocab_size, max_seq_length, num_classes)
        inputs = Input(shape=(max_seq_length,))
        embedding_layer = Embedding(input_dim=vocab_size, output_dim=256, input_length=max_seq_length)(inputs)
        lstm_layer = Bidirectional(LSTM(128, return_sequences=True))(embedding_layer)
        attention_layer = Attention()([lstm_layer, lstm_layer])
        lstm_with_attention = Bidirectional(LSTM(128))(attention_layer)
        
        dense_layer = Dense(64, activation='relu')(lstm_with_attention)
        dropout_layer = Dropout(0.2)(dense_layer)
        output_layer = Dense(num_classes, activation='softmax')(dropout_layer)

        self.model = Model(inputs=inputs, outputs=output_layer)
        self.model.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

class GloveMaxPoolingBiLSTM(BaseModel):
    """
    A class representing a Bidirectional LSTM model with GloVe embeddings and max pooling.
    """
    def __init__(self, vocab_size, max_seq_length, num_classes, embedding_matrix, is_trainable=False):
        super().__init__(vocab_size, max_seq_length, num_classes)
        base_model = Sequential([
            Embedding(vocab_size, embedding_matrix.shape[1], weights=[embedding_matrix], input_length=max_seq_length, trainable=is_trainable),
            Bidirectional(LSTM(128, return_sequences=True)),
            Bidirectional(LSTM(128, return_sequences=True))
        ])

        max_pool = GlobalMaxPooling1D()(base_model.output)
        dense = Dense(64, activation='relu')(max_pool)
        dropout = Dropout(0.2)(dense)
        output = Dense(num_classes, activation='softmax')(dropout)

        self.model = Model(inputs=base_model.input, outputs=output)
        self.model.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])

class BertTinyMaxPoolingBiLSTM(BaseModel):
    """
    A class representing a BERT Tiny model with Bidirectional LSTM and Max Pooling layers.
    """
    def __init__(self, max_seq_length=128, num_classes=0, bert_model_name='google/bert_uncased_L-2_H-128_A-2', is_trainable=False):
        super().__init__(None, max_seq_length, num_classes)

        # Load BERT tiny model
        bert_model = TFBertModel.from_pretrained(bert_model_name, trainable=is_trainable)

        input_ids = Input(shape=(max_seq_length,), dtype='int32')
        attention_mask = Input(shape=(max_seq_length,), dtype='int32')

        bert_output = bert_model(input_ids, attention_mask=attention_mask)[0]

        lstm_layer = Bidirectional(LSTM(128, return_sequences=True))(bert_output)
        lstm_layer = Bidirectional(LSTM(128, return_sequences=True))(lstm_layer)
        max_pool_layer = GlobalMaxPooling1D()(lstm_layer)

        dense_layer = Dense(64, activation='relu')(max_pool_layer)
        dropout_layer = Dropout(0.2)(dense_layer)
        output_layer = Dense(num_classes, activation='softmax')(dropout_layer)

        self.model = Model(inputs=[input_ids, attention_mask], outputs=output_layer)
        self.model.compile(optimizer=Adam(learning_rate=0.01), loss='categorical_crossentropy', metrics=['accuracy'])


class Experimental(BaseModel):
    """
    An experimental class representing dummy for testing.
    """
    def __init__(self, vocab_size, max_seq_length, num_classes):
        super().__init__(vocab_size, max_seq_length, num_classes)
        base_model = Sequential([
            Embedding(input_dim=vocab_size, output_dim=256, input_length=max_seq_length),
            Bidirectional(LSTM(128, return_sequences=True)),
            Bidirectional(LSTM(128, return_sequences=True))
        ])

        max_pool = GlobalMaxPooling1D()(base_model.output)
        dense = Dense(64, activation='relu')(max_pool)
        dropout = Dropout(0.2)(dense)
        output = Dense(num_classes, activation='softmax')(dropout)

        self.model = Model(inputs=base_model.input, outputs=output)
        self.model.compile(optimizer=Adam(lr=0.01), loss='categorical_crossentropy', metrics=['accuracy'])
