# Standard library imports
import json

# Local imports
from .preprocess import DataPreprocessor, BertTinyDataPreprocessor, GloVeEmbeddings
from .lstm_models import (
    BaseLSTM, BiLSTM, RegularizedBiLSTM, AvgPoolingBiLSTM, MaxPoolingBiLSTM,
    AttentionBiLSTM, GloveMaxPoolingBiLSTM, BertTinyMaxPoolingBiLSTM, Experimental
)

# Third-party imports
import ipywidgets as widgets
from IPython.display import display, clear_output


class LSTMTrainer:
    """
    A class for training LSTM models.
    """

    def __init__(self, model_name, val_split=0.1, epochs=10, batch_size=64, is_trainable=False):
        """
        Initialize the LSTMTrainer.

        Args:
            model_name (str): The name of the LSTM model to train.
            val_split (float): The proportion of the training data to use for validation.
            epochs (int): The number of epochs to train the model.
            batch_size (int): The batch size to use during training.
            is_trainable (bool): Whether the embeddings should be trainable.
        """
        self.model_name = model_name
        self.val_split = val_split
        self.epochs = epochs
        self.batch_size = batch_size
        self.is_trainable = is_trainable

        # Load the training and testing data
        with open('data/full/train.json', 'r') as file:
            self.train_data = json.load(file)
        with open('data/full/test.json', 'r') as file:
            self.test_data = json.load(file)

        # Adjust data and parameters based on the selected model
        if model_name == 'BertTinyMaxPoolingBiLSTM':
            bert_preprocessor = BertTinyDataPreprocessor()
            self.train_data, self.train_labels, self.test_data, self.test_labels = bert_preprocessor.preprocess(
                self.train_data, self.test_data
            )
            self.num_classes = bert_preprocessor.get_num_classes()
            self.max_seq_length = bert_preprocessor.max_seq_length
        else:
            preprocessor = DataPreprocessor()
            self.train_padded, self.train_labels, self.test_padded, self.test_labels = preprocessor.preprocess(
                self.train_data, self.test_data
            )
            self.vocab_size = preprocessor.get_vocab_size()
            self.num_classes = preprocessor.get_num_classes()
            self.max_seq_length = preprocessor.get_max_seq_length()
            if model_name == 'GloveMaxPoolingBiLSTM':
                glove_embeddings = GloVeEmbeddings('./utils/glove.6B.50d.txt', 50)
                self.embedding_matrix = glove_embeddings.create_embedding_matrix(preprocessor.tokenizer.word_index)

        self.setup_model()

    def setup_model(self):
        """
        Set up the LSTM model based on the selected model name.
        """
        if self.model_name == 'BaseLSTM':
            self.model = BaseLSTM(self.vocab_size, self.max_seq_length, self.num_classes)
        elif self.model_name == 'BiLSTM':
            self.model = BiLSTM(self.vocab_size, self.max_seq_length, self.num_classes)
        elif self.model_name == 'RegularizedBiLSTM':
            self.model = RegularizedBiLSTM(self.vocab_size, self.max_seq_length, self.num_classes)
        elif self.model_name == 'MaxPoolingBiLSTM':
            self.model = MaxPoolingBiLSTM(self.vocab_size, self.max_seq_length, self.num_classes)
        elif self.model_name == 'AvgPoolingBiLSTM':
            self.model = AvgPoolingBiLSTM(self.vocab_size, self.max_seq_length, self.num_classes)
        elif self.model_name == 'AttentionBiLSTM':
            self.model = AttentionBiLSTM(self.vocab_size, self.max_seq_length, self.num_classes)
        elif self.model_name == 'GloveMaxPoolingBiLSTM':
            self.model = GloveMaxPoolingBiLSTM(
                self.vocab_size, self.max_seq_length, self.num_classes, self.embedding_matrix, is_trainable=self.is_trainable
            )
        elif self.model_name == 'BertTinyMaxPoolingBiLSTM':
            self.model = BertTinyMaxPoolingBiLSTM(self.max_seq_length, self.num_classes, is_trainable=self.is_trainable)
        elif self.model_name == 'Experimental':
            self.model = Experimental(self.vocab_size, self.max_seq_length, self.num_classes)
        else:
            raise ValueError("Invalid model selection")

    def train(self):
        """
        Train the LSTM model.
        """
        print(
            f"\nTraining {self.model_name} with validation split: {self.val_split}, epochs: {self.epochs}, "
            f"batch size: {self.batch_size}, trainable: {self.is_trainable}"
        )

        if self.model_name == 'BertTinyMaxPoolingBiLSTM':
            print("\n")
            print(50 * "=" + "TRAINING" + 50 * "=")
            self.model.train(
                self.train_data, self.train_labels, validation_split=self.val_split,
                epochs=self.epochs, batch_size=self.batch_size
            )
            print("\n")
            print(50 * "=" + "TESTING" + 50 * "=")
            self.model.evaluate(self.test_data, self.test_labels)
        else:
            print("\n")
            print(50 * "=" + "TRAINING" + 50 * "=")
            self.model.train(
                self.train_padded, self.train_labels, validation_split=self.val_split,
                epochs=self.epochs, batch_size=self.batch_size
            )
            print("\n")
            print(50 * "=" + "TESTING" + 50 * "=")
            self.model.evaluate(self.test_padded, self.test_labels)

        print("Training complete.")


class LSTMTrainGUI:
    """
    A graphical user interface for training LSTM models.
    """

    def __init__(self):
        self.setup_widgets()
        self.display_widgets()

    def setup_widgets(self):
        """
        Set up the GUI widgets.
        """
        self.model_name_widget = widgets.Dropdown(
            options=[
                'BaseLSTM', 'BiLSTM', 'RegularizedBiLSTM', 'AvgPoolingBiLSTM', 'MaxPoolingBiLSTM',
                'AttentionBiLSTM', 'GloveMaxPoolingBiLSTM', 'BertTinyMaxPoolingBiLSTM', 'Experimental'
            ],
            value='BaseLSTM',
            description='Model:',
        )

        self.val_split_widget = widgets.FloatSlider(
            value=0.1, min=0, max=0.5, step=0.05, description='Val Split:'
        )
        self.epochs_widget = widgets.IntSlider(
            value=10, min=1, max=1000, step=1, description='Epochs:'
        )
        self.batch_size_widget = widgets.IntSlider(
            value=64, min=1, max=4096, step=1, description='Batch Size:'
        )
        self.is_trainable_widget = widgets.Checkbox(
            value=False, description='Trainable Embeddings'
        )

        self.train_button = widgets.Button(description='Train Model')
        self.train_button.on_click(self.on_train_button_clicked)

    def display_widgets(self):
        """
        Display the GUI widgets.
        """
        display(widgets.VBox([
            self.model_name_widget,
            self.val_split_widget,
            self.epochs_widget,
            self.batch_size_widget,
            self.is_trainable_widget,
            self.train_button
        ]))

    def on_train_button_clicked(self, _):
        """
        Callback function for the "Train Model" button.
        """
        clear_output(wait=True)

        trainer = LSTMTrainer(
            model_name=self.model_name_widget.value,
            val_split=self.val_split_widget.value,
            epochs=self.epochs_widget.value,
            batch_size=self.batch_size_widget.value,
            is_trainable=self.is_trainable_widget.value
        )
        trainer.train()
