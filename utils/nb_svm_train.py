# Third-party imports
import ipywidgets as widgets
from IPython.display import display, clear_output

# Local imports
from .nb_svm_models import NaiveBayesModels as NBM
from .nb_svm_models import SVMs as SVM
from .helpers import readData, save_model


class NBSVMTrainGUI:
    """
    A graphical user interface for training Naive Bayes and SVM models.
    """

    def __init__(self, test_data_path, train_data_path):
        """
        Initialize the NBSVMTrainGUI.

        Args:
            test_data_path (str): The path to the test data file.
            train_data_path (str): The path to the train data file.
        """
        self.testDF = readData(test_data_path)
        self.trainDF = readData(train_data_path)

        self.setup_widgets()

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
            options=['GaussianNB', 'MultinomialNB', 'BernoulliNB', 'ComplementNB'],
            value='GaussianNB',
            description='NB Model:',
        )

        self.feature_type_widget = widgets.Dropdown(
            options=['BoW', 'TFIDF', 'BERT-tiny', 'BERT-small'],
            value='BoW',
            description='Feature Type:',
        )

        self.train_button = widgets.Button(description='Train Model')
        self.train_button.on_click(self.train_model)

        self.widget_box = widgets.VBox([
            self.model_type_widget,
            self.nb_type_widget,
            self.feature_type_widget,
            self.train_button,
        ])
        display(self.widget_box)

    def train_model(self, _):
        """
        Train the selected model based on the user's choices.
        """
        clear_output(wait=True)

        model_type = self.model_type_widget.value
        if model_type == 'Naive Bayes':
            self.train_naive_bayes()
        elif model_type == 'SVM':
            self.train_svm()

        display(self.widget_box)

    def train_naive_bayes(self):
        """
        Train the Naive Bayes model based on the selected options.
        """
        nb_type = self.nb_type_widget.value
        feature_type = self.feature_type_widget.value

        if feature_type in ['BERT-small', 'BERT-tiny']:
            bert_model_name = 'prajjwal1/bert-small' if feature_type == 'BERT-small' else 'prajjwal1/bert-tiny'
        else:
            bert_model_name = None

        nbm = NBM(
            testDF=self.testDF,
            trainDF=self.trainDF,
            bert=feature_type in ['BERT-small', 'BERT-tiny'],
            emb_model_name=bert_model_name
        )
        nbm.prepare_dataset(
            use_tfidf=feature_type == 'TFIDF',
            use_bert=feature_type in ['BERT-small', 'BERT-tiny']
        )

        if nb_type == 'GaussianNB':
            model_res = nbm.GaussianNB()
        elif nb_type == 'MultinomialNB':
            model_res = nbm.MultinomialNB()
        elif nb_type == 'BernoulliNB':
            model_res = nbm.BernoulliNB()
        elif nb_type == 'ComplementNB':
            model_res = nbm.ComplementNB()

        save_model(model_res['model'], f'NB-{nb_type}-{feature_type}')

    def train_svm(self):
        """
        Train the SVM model based on the selected options.
        """
        feature_type = self.feature_type_widget.value

        if feature_type in ['BERT-small', 'BERT-tiny']:
            bert_model_name = 'prajjwal1/bert-small' if feature_type == 'BERT-small' else 'prajjwal1/bert-tiny'
        else:
            bert_model_name = None

        svm = SVM(testDF=self.testDF, trainDF=self.trainDF, emb_model_name=bert_model_name)
        svm.prepare_dataset()
        res = svm.fit_and_score()
        save_model(res, name=f'SVC-{feature_type}')
