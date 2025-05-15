# Third-party imports
import ipywidgets as widgets
import joblib
import numpy as np
import torch
from IPython.display import display, clear_output
from sklearn.preprocessing import LabelEncoder

# Local imports
from .nb_svm_models import NaiveBayesModels as NBM
from .nb_svm_models import SVMs as SVM
from .helpers import readData


class NBSVMInferenceGUI:
    """
    A graphical user interface for performing inference with Naive Bayes and SVM models.
    """

    def __init__(self, test_data_path, train_data_path):
        """
        Initialize the NBSVMInferenceGUI.

        Args:
            test_data_path (str): The path to the test data file.
            train_data_path (str): The path to the train data file.
        """
        self.testDF = readData(test_data_path)
        self.trainDF = readData(train_data_path)
        self.prev_model_type = None
        self.prev_nb_type = None
        self.prev_feature_type = None
        self.model_cache = {}
        self.setup_widgets()
        self.display_widgets()

    def setup_widgets(self):
        """
        Set up the GUI widgets.
        """
        self.model_type_widget = widgets.Dropdown(
            options=['Naive Bayes', 'SVM'],
            value='Naive Bayes',
            description='Model Type:',
        )
        self.nb_type_widget = widgets.Dropdown(
            options=['gaussian', 'multinomial', 'bernoulli', 'complement'],
            value='gaussian',
            description='NB Model:',
        )
        self.feature_type_widget = widgets.Dropdown(
            options=['BoW', 'TFIDF', 'BERT-tiny', 'BERT-small'],
            value='BoW',
            description='Feature Type:',
        )

        self.input_text = widgets.Textarea(
            value='This is an <e1>example</e1> of a properly formatted input <e2>sentence</e2> between relations.',
            placeholder='Enter a sentence',
            description='Input Sentence:',
            disabled=False
        )

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

        self.output_widget = widgets.Output()

    def display_widgets(self):
        """
        Display the GUI widgets.
        """
        display(widgets.VBox([
            self.model_type_widget,
            self.nb_type_widget,
            self.feature_type_widget,
            self.input_text,
            self.get_predictions_button,
            self.reset_output_button,
            self.output_widget
        ]))

    def get_predictions(self, _):
        """
        Callback function for the "Get Predictions" button.
        """
        with self.output_widget:
            clear_output(wait=True)
            text = self.input_text.value
            model_type = self.model_type_widget.value
            nb_type = self.nb_type_widget.value
            feature_type = self.feature_type_widget.value

            if model_type != self.prev_model_type or nb_type != self.prev_nb_type or feature_type != self.prev_feature_type:
                self.prev_model_type = model_type
                self.prev_nb_type = nb_type
                self.prev_feature_type = feature_type

                cache_key = f"{model_type}-{nb_type}-{feature_type}"
                if cache_key in self.model_cache:
                    inference = self.model_cache[cache_key]
                else:
                    inference = NBSVMInference(self.testDF, self.trainDF, model_type, nb_type, feature_type)
                    self.model_cache[cache_key] = inference
            else:
                inference = self.model_cache[f"{model_type}-{nb_type}-{feature_type}"]

            if model_type == 'Naive Bayes':
                predictions, confidences = inference.predict(text)
                print("Top 3 Predicted Labels:")
                for label, conf in zip(predictions, confidences):
                    print(f"{label}: {conf:.3f}")
            else:
                predictions = inference.predict(text)
                print("Top Predicted Label:")
                for label in predictions:
                    print(f"{label}")

    def reset_output(self, _):
        """
        Callback function for the "Reset Output" button.
        """
        self.output_widget.clear_output()


class NBSVMInference:
    """
    A class for performing inference with Naive Bayes and SVM models.
    """

    def __init__(self, testDF, trainDF, model_type, nb_type, feature_type):
        """
        Initialize the NBSVMInference.

        Args:
            testDF (pd.DataFrame): The test data DataFrame.
            trainDF (pd.DataFrame): The train data DataFrame.
            model_type (str): The type of model to use (Naive Bayes or SVM).
            nb_type (str): The type of Naive Bayes model to use (gaussian, multinomial, bernoulli, complement).
            feature_type (str): The type of features to use (BoW, TFIDF, BERT-tiny, BERT-small).
        """
        self.testDF = testDF
        self.trainDF = trainDF
        self.model_type = model_type
        self.nb_type = nb_type
        self.feature_type = feature_type
        self.le = LabelEncoder()
        self.le.fit(self.trainDF['relation'])

        if model_type == 'Naive Bayes':
            self.model = joblib.load(f'checkpoints/PRETRAINED/nb_svm/NB-{nb_type}-{feature_type}-model.joblib')
        elif model_type == 'SVM':
            self.model = joblib.load(f'checkpoints/PRETRAINED/nb_svm/SVC-{feature_type}-model.joblib')['model']

    def predict(self, text):
        """
        Predict the labels for the given text.

        Args:
            text (str): The input text.

        Returns:
            tuple or np.ndarray: The predicted labels and confidences (for Naive Bayes) or the predicted labels (for SVM).
        """
        if self.model_type == 'Naive Bayes':
            nbm = NBM(testDF=self.testDF, trainDF=self.trainDF, bert=self.feature_type in ['BERT-small', 'BERT-tiny'])
            nbm.prepare_dataset(use_tfidf=self.feature_type == 'TFIDF', use_bert=self.feature_type in ['BERT-small', 'BERT-tiny'])
            test_data = self._vectorize_text(self.model_type, nbm, [text])
        elif self.model_type == 'SVM':
            svm = SVM(testDF=self.testDF, trainDF=self.trainDF)
            svm.prepare_dataset()
            test_data = self._vectorize_text(self.model_type, svm, [text])

        if self.model_type == 'Naive Bayes':
            probabilities = self.model.predict_proba(test_data)[0]
            top_indices = probabilities.argsort()[-3:][::-1]
            top_predictions = self.model.classes_[top_indices]
            top_confidences = probabilities[top_indices]
            return top_predictions, top_confidences
        else:
            pred = self.model.predict(test_data)
            top_predictions = self.le.inverse_transform(pred)
            return top_predictions

    def _vectorize_text(self, model_type, model_instance, texts):
        """
        Vectorize the input texts using the specified feature type.

        Args:
            model_type (str): The type of model (Naive Bayes or SVM).
            model_instance (NBM or SVM): The model instance.
            texts (list): The list of input texts.

        Returns:
            np.ndarray: The vectorized texts.
        """
        if self.feature_type == 'BoW':
            return model_instance.bow.transform(texts).toarray()
        elif self.feature_type == 'TFIDF':
            return model_instance.tfidf.transform(texts).toarray()
        else:  # BERT or BERTs
            embeddings = []
            for text in texts:
                if model_type == 'Naive Bayes':
                    embedds = model_instance._NaiveBayesModels__get_bert_embeddings(text)
                else:
                    embedds = model_instance._SVMs__get_bert_embeddings(text)
                embeddings.append(torch.mean(embedds, dim=0).numpy())
            return np.array(embeddings)