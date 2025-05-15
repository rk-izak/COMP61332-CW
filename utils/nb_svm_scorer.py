# Third-party imports
import ipywidgets as widgets
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display, clear_output
from sklearn.metrics import confusion_matrix, classification_report

# Local imports
from .nb_svm_models import NaiveBayesModels as NBM
from .nb_svm_models import SVMs as SVM
from .helpers import readData


class NBSVMScorerGUI:
    """
    A graphical user interface for scoring Naive Bayes and SVM models.
    """

    def __init__(self, test_data_path, train_data_path):
        """
        Initialize the NBSVMScorerGUI.

        Args:
            test_data_path (str): The path to the test data file.
            train_data_path (str): The path to the train data file.
        """
        self.testDF = readData(test_data_path)
        self.trainDF = readData(train_data_path)
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

        self.score_button = widgets.Button(description='Score Model')
        self.score_button.on_click(self.on_score_button_clicked)

        self.output_widget = widgets.Output()

    def display_widgets(self):
        """
        Display the GUI widgets.
        """
        display(widgets.VBox([
            self.model_type_widget,
            self.nb_type_widget,
            self.feature_type_widget,
            self.score_button,
            self.output_widget
        ]))

    def on_score_button_clicked(self, _):
        """
        Callback function for the "Score Model" button.
        """
        with self.output_widget:
            clear_output(wait=True)
            scorer = NBSVMScorer(
                self.testDF,
                self.trainDF,
                self.model_type_widget.value,
                self.nb_type_widget.value,
                self.feature_type_widget.value
            )
            scorer.score()


class NBSVMScorer:
    """
    A class for scoring Naive Bayes and SVM models.
    """

    def __init__(self, testDF, trainDF, model_type, nb_type, feature_type):
        """
        Initialize the NBSVMScorer.

        Args:
            testDF (pd.DataFrame): The test data DataFrame.
            trainDF (pd.DataFrame): The train data DataFrame.
            model_type (str): The type of model to score (Naive Bayes or SVM).
            nb_type (str): The type of Naive Bayes model (gaussian, multinomial, bernoulli, complement).
            feature_type (str): The type of features used (BoW, TFIDF, BERT-tiny, BERT-small).
        """
        self.testDF = testDF
        self.trainDF = trainDF
        self.model_type = model_type
        self.nb_type = nb_type
        self.feature_type = feature_type

        if model_type == 'Naive Bayes':
            self.model = joblib.load(f'checkpoints/PRETRAINED/nb_svm/NB-{nb_type}-{feature_type}-model.joblib')
        elif model_type == 'SVM':
            self.model = joblib.load(f'checkpoints/PRETRAINED/nb_svm/SVC-{feature_type}-model.joblib')['model']

    def score(self):
        """
        Score the model and display the classification report and confusion matrix.
        """
        if self.feature_type in ['BERT-small', 'BERT-tiny']:
            bert_model_name = 'prajjwal1/bert-small' if self.feature_type == 'BERT-small' else 'prajjwal1/bert-tiny'
        else:
            bert_model_name = None

        if self.model_type == 'Naive Bayes':
            nbm = NBM(
                testDF=self.testDF,
                trainDF=self.trainDF,
                bert=self.feature_type in ['BERT-small', 'BERT-tiny'],
                emb_model_name=bert_model_name
            )
            nbm.prepare_dataset(
                use_tfidf=self.feature_type == 'TFIDF',
                use_bert=self.feature_type in ['BERT-small', 'BERT-tiny']
            )
            test_data, test_labels = nbm.X_test, nbm.y_test
        elif self.model_type == 'SVM':
            svm = SVM(testDF=self.testDF, trainDF=self.trainDF, emb_model_name=bert_model_name)
            svm.prepare_dataset()
            test_data, test_labels = svm.X_test, svm.y_test

        predictions = self.model.predict(test_data)
        predicted_labels = predictions

        print("\n")
        print(40 * "=" + f" SCORES FOR {self.model_type} - {self.nb_type if self.model_type == 'Naive Bayes' else ''} ({self.feature_type}): " + 40 * "=")
        print("\n")
        report = classification_report(test_labels, predicted_labels)
        print(report)

        class_names = list(set(test_labels))
        matrix = confusion_matrix(test_labels, predicted_labels)
        plt.figure(figsize=(10, 7))
        sns.heatmap(matrix, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title(f'Confusion Matrix for {self.model_type} - {self.nb_type if self.model_type == "Naive Bayes" else ""} ({self.feature_type})')
        plt.show()
