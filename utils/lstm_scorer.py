# Standard library imports
import json

# Third-party imports
import ipywidgets as widgets
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from IPython.display import display, clear_output
from keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from transformers import TFBertModel

# Local imports
from .preprocess import DataPreprocessor, BertTinyDataPreprocessor, GloVeEmbeddings
from .lstm_models import (
    BaseLSTM, BiLSTM, RegularizedBiLSTM, AvgPoolingBiLSTM, MaxPoolingBiLSTM,
    AttentionBiLSTM, GloveMaxPoolingBiLSTM, BertTinyMaxPoolingBiLSTM, Experimental
)


class LSTMScorer:
    """
    A class for scoring LSTM models.
    """

    def __init__(self, model_name):
        """
        Initialize the LSTMScorer.

        Args:
            model_name (str): The name of the LSTM model to score.
        """
        self.model_name = model_name

        # Load the training and testing data
        with open('data/full/train.json', 'r') as file:
            self.train_data = json.load(file)
        with open('data/full/test.json', 'r') as file:
            self.test_data = json.load(file)

        # Preprocess the data based on the model type
        if self.model_name == 'BertTinyMaxPoolingBiLSTM':
            self.preprocessor = BertTinyDataPreprocessor()
        else:
            self.preprocessor = DataPreprocessor()

        self.train_data, self.train_labels, self.test_data, self.test_labels = self.preprocessor.preprocess(
            self.train_data, self.test_data
        )
        self.vocab_size = self.preprocessor.get_vocab_size()
        self.num_classes = self.preprocessor.get_num_classes()
        self.max_seq_length = self.preprocessor.get_max_seq_length()

        # Load the GloVe embeddings if using the GloveMaxPoolingBiLSTM model
        if self.model_name == 'GloveMaxPoolingBiLSTM':
            glove_embeddings = GloVeEmbeddings('./utils/glove.6B.50d.txt', 50)
            self.embedding_matrix = glove_embeddings.create_embedding_matrix(self.preprocessor.tokenizer.word_index)

        # Load the pre-trained model based on the model name
        if self.model_name == 'BaseLSTM':
            self.model = BaseLSTM(self.vocab_size, self.max_seq_length, self.num_classes)
            self.model.model = load_model('./checkpoints/PRETRAINED/lstm/BaseLSTM_model.h5')
        elif self.model_name == 'BiLSTM':
            self.model = BiLSTM(self.vocab_size, self.max_seq_length, self.num_classes)
            self.model.model = load_model('./checkpoints/PRETRAINED/lstm/BiLSTM_model.h5')
        elif self.model_name == 'RegularizedBiLSTM':
            self.model = RegularizedBiLSTM(self.vocab_size, self.max_seq_length, self.num_classes)
            self.model.model = load_model('./checkpoints/PRETRAINED/lstm/RegularizedBiLSTM_model.h5')
        elif self.model_name == 'MaxPoolingBiLSTM':
            self.model = MaxPoolingBiLSTM(self.vocab_size, self.max_seq_length, self.num_classes)
            self.model.model = load_model('./checkpoints/PRETRAINED/lstm/MaxPoolingBiLSTM_model.h5')
        elif self.model_name == 'AvgPoolingBiLSTM':
            self.model = AvgPoolingBiLSTM(self.vocab_size, self.max_seq_length, self.num_classes)
            self.model.model = load_model('./checkpoints/PRETRAINED/lstm/AvgPoolingBiLSTM_model.h5')
        elif self.model_name == 'AttentionBiLSTM':
            self.model = AttentionBiLSTM(self.vocab_size, self.max_seq_length, self.num_classes)
            self.model.model = load_model('./checkpoints/PRETRAINED/lstm/AttentionBiLSTM_model.h5')
        elif self.model_name == 'GloveMaxPoolingBiLSTM':
            self.model = GloveMaxPoolingBiLSTM(
                self.vocab_size, self.max_seq_length, self.num_classes, self.embedding_matrix, is_trainable=False
            )
            self.model.model = load_model('./checkpoints/PRETRAINED/lstm/GloveMaxPoolingBiLSTM_model.h5')
        elif self.model_name == 'BertTinyMaxPoolingBiLSTM':
            self.model = BertTinyMaxPoolingBiLSTM(self.max_seq_length, self.num_classes, is_trainable=False)
            self.model.model = load_model(
                f'./checkpoints/PRETRAINED/lstm/{self.model_name}_model.h5',
                custom_objects={'TFBertModel': TFBertModel}
            )
        elif self.model_name == 'Experimental':
            self.model = Experimental(self.vocab_size, self.max_seq_length, self.num_classes)
            self.model.model = load_model('./checkpoints/PRETRAINED/lstm/Experimental_model.h5')
        else:
            raise ValueError("Invalid model selection")

    def score(self):
        """
        Score the model and display the classification report and confusion matrix.
        """
        class_names = self.preprocessor.get_class_names()
        predictions = self.model.model.predict(self.test_data)
        predicted_labels = np.argmax(predictions, axis=1)
        true_labels = np.argmax(self.test_labels, axis=1)

        print("\n")
        print(40 * "=" + f" SCORES FOR {self.model_name}: " + 40 * "=")
        print("\n")
        report = classification_report(true_labels, predicted_labels, target_names=class_names, zero_division=0)
        print(report)

        matrix = confusion_matrix(true_labels, predicted_labels)
        plt.figure(figsize=(10, 7))
        sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title(f'Confusion Matrix for {self.model_name}')
        plt.show()

    def predict(self, sentence):
        """
        Predict the top 3 labels for a given sentence.

        Args:
            sentence (str): The input sentence.

        Returns:
            list: A list of tuples containing the top 3 predicted labels and their probabilities.
        """
        if self.model_name == 'BertTinyMaxPoolingBiLSTM':
            input_ids, attention_mask = self.preprocessor.preprocess_single_sentence(sentence)
            predictions = self.model.model.predict([input_ids, attention_mask])
        else:
            padded_sequence = self.preprocessor.preprocess_single_sentence(sentence)
            predictions = self.model.model.predict(padded_sequence)

        class_names = self.preprocessor.get_class_names()
        top_predictions = np.argsort(predictions[0])[::-1][:3]
        top_probs = predictions[0][top_predictions]
        top_predictions = [class_names[pred] for pred in top_predictions]

        return list(zip(top_predictions, top_probs))


class LSTMScorerGUI:
    """
    A graphical user interface for scoring LSTM models.
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

        self.score_button = widgets.Button(description='Score Model')
        self.score_button.on_click(self.on_score_button_clicked)

        self.output_widget = widgets.Output()

    def display_widgets(self):
        """
        Display the GUI widgets.
        """
        display(widgets.VBox([self.model_name_widget, self.score_button, self.output_widget]))

    def on_score_button_clicked(self, _):
        """
        Callback function for the "Score Model" button.
        """
        with self.output_widget:
            clear_output(wait=True)
            scorer = LSTMScorer(model_name=self.model_name_widget.value)
            scorer.score()
            plt.show()


class LSTMInferenceGUI:
    """
    A graphical user interface for performing inference with LSTM models.
    """

    def __init__(self):
        self.prev_model_name = None
        self.predictor_cache = {}  # Dictionary to store loaded predictors
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

        self.input_text = widgets.Textarea(
            value='This is an <e1>example</e1> of a properly formatted input <e2>sentence</e2> between relations.',
            placeholder='Enter a sentence',
            description='Input Sentence:',
            disabled=False
        )

        self.output_text = widgets.Output()

        self.get_predictions_button = widgets.Button(
            description='Get Predictions',
            disabled=False,
            button_style='success'
        )
        self.get_predictions_button.on_click(self.get_predictions)

        self.reset_output_button = widgets.Button(
            description='Reset Output',
            disabled=False,
            button_style='danger'
        )
        self.reset_output_button.on_click(self.reset_output)

    def display_widgets(self):
        """
        Display the GUI widgets.
        """
        display(widgets.VBox([
            self.model_name_widget,
            self.input_text,
            self.get_predictions_button,
            self.reset_output_button,
            self.output_text
        ]))

    def get_predictions(self, _):
        """
        Callback function for the "Get Predictions" button.
        """
        with self.output_text:
            self.output_text.clear_output(wait=True)
            sentence = self.input_text.value
            model_name = self.model_name_widget.value

            if model_name != self.prev_model_name:
                self.prev_model_name = model_name
                if model_name in self.predictor_cache:
                    self.predictor = self.predictor_cache[model_name]
                else:
                    self.predictor = LSTMScorer(model_name)
                    self.predictor_cache[model_name] = self.predictor

            top_predictions = self.predictor.predict(sentence)
            print(f"Top 3 Predicted Labels:")
            for label, conf in top_predictions:
                print(f"{label}: {conf:.3f}")

    def reset_output(self, _):
        """
        Callback function for the "Reset Output" button.
        """
        with self.output_text:
            self.output_text.clear_output()