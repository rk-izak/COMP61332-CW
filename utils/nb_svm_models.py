# Third-party imports
import torch
from sklearn.feature_extraction.text import CountVectorizer as cv
from sklearn.feature_extraction.text import TfidfVectorizer as tf_idf
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from transformers import BertModel, BertTokenizerFast

# Basic imports
import pandas as pd


class NaiveBayesModels:
    """
    A class for training and evaluating Naive Bayes models.
    """

    special_tokens = ["[E1]", "[/E1]", "[E2]", "[/E2]"]
    X_train, y_train, X_test, y_test = None, None, None, None
    bert_model, bert_tokenizer = None, None
    trainDF, testDF = None, None
    tfidf = tf_idf()
    bow = cv()
    emb_model_name = ""
    scaler = MinMaxScaler()

    info = {"embeddings": "None", "model": "None"}

    def __init__(self, trainDF, testDF, bert=False, emb_model_name="prajjwal1/bert-tiny"):
        """
        Initialize the NaiveBayesModels class.

        Args:
            trainDF (pd.DataFrame): The training data DataFrame.
            testDF (pd.DataFrame): The testing data DataFrame.
            bert (bool): Whether to use BERT embeddings. Defaults to False.
            emb_model_name (str): The name of the BERT model to load. Defaults to "prajjwal1/bert-tiny".
        """
        if bert:
            self.emb_model_name = emb_model_name
            self._load_bert(emb_model_name)
        self.trainDF = trainDF
        self.testDF = testDF

    def GaussianNB(self):
        """
        Train and evaluate a Gaussian Naive Bayes model.

        Returns:
            dict: A dictionary containing the fitted model instance and metadata.
        """
        if self._check_basic_setup():
            return self._fit_and_score(GaussianNB())
        return ValueError("Data not loaded")

    def MultinomialNB(self):
        """
        Train and evaluate a Multinomial Naive Bayes model.

        Returns:
            dict: A dictionary containing the fitted model instance and metadata.
        """
        if self._check_basic_setup(check_positive=True):
            return self._fit_and_score(MultinomialNB())
        return ValueError("Data not loaded")

    def BernoulliNB(self):
        """
        Train and evaluate a Bernoulli Naive Bayes model.

        Returns:
            dict: A dictionary containing the fitted model instance and metadata.
        """
        if self._check_basic_setup():
            return self._fit_and_score(BernoulliNB())
        return ValueError("Data not loaded")

    def ComplementNB(self):
        """
        Train and evaluate a Complement Naive Bayes model.

        Returns:
            dict: A dictionary containing the fitted model instance and metadata.
        """
        if self._check_basic_setup(check_positive=True):
            return self._fit_and_score(ComplementNB())
        return ValueError("Data not loaded")

    def _fit_and_score(self, model):
        """
        Fits the provided model on the training data and scores it on the test data.

        Args:
            model (sklearn.base.BaseEstimator): The sklearn model to fit and evaluate.

        Returns:
            dict: A dictionary containing the fitted model instance and metadata.
        """
        self.info["model"] = model.__class__.__name__
        print("\n")
        print(50 * "=" + "TRAINING" + 50 * "=")
        model.fit(self.X_train, self.y_train)

        # Calculate train accuracy
        train_accuracy = model.score(self.X_train, self.y_train)
        print(f"Train Accuracy: {train_accuracy:.2f}")

        print("\n")
        print(50 * "=" + "TESTING" + 50 * "=")
        # Calculate test accuracy
        test_accuracy = model.score(self.X_test, self.y_test)
        print(f"Test Accuracy: {test_accuracy:.2f}")
        print("\nTraining complete.")

        return {"model": model, "info": self.info}

    def _check_basic_setup(self, check_positive=False):
        """
        Check if the basic setup is complete.

        Args:
            check_positive (bool): Whether to check for positive values when using BERT embeddings. Defaults to False.

        Returns:
            bool: True if the basic setup is complete, False otherwise.
        """
        if self.X_train is None or self.y_train is None or self.X_test is None or self.y_test is None:
            return False
        if self.bert_model is not None and check_positive:
            # Min-Max scaling of BERT embeddings
            self.X_train = self.scaler.fit_transform(self.X_train)
            self.X_test = self.scaler.transform(self.X_test)
        return True

    def _load_bert(self, model_name):
        """
        Load the BERT model and tokenizer.

        Args:
            model_name (str): The name of the BERT model to load.
        """
        self.bert_tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.bert_model = BertModel.from_pretrained(model_name)
        self.bert_tokenizer.add_special_tokens({"additional_special_tokens": self.special_tokens})
        self.bert_model.resize_token_embeddings(len(self.bert_tokenizer))

    def _get_bert_embeddings(self, sentence):
        """
        Get BERT embeddings for a sentence.

        Args:
            sentence (str): The input sentence.

        Returns:
            torch.Tensor: The BERT embeddings for the sentence.
        """
        # Tokenize the sentence
        sentence_tokens = self.bert_tokenizer.tokenize(sentence)

        # Convert the sentence tokens to input IDs
        input_ids = self.bert_tokenizer.convert_tokens_to_ids(sentence_tokens)

        # Create the input tensor
        input_tensor = torch.tensor([input_ids])

        # Get BERT embeddings for the sentence
        with torch.no_grad():
            outputs = self.bert_model(input_tensor)
            sentence_embedding = outputs.last_hidden_state.squeeze(0)

        return sentence_embedding

    def prepare_dataset(self, use_bert=False, use_tfidf=False):
        """
        Prepare the dataset for model training and evaluation by generating embeddings for the text data.

        Args:
            use_bert (bool): Whether to use BERT embeddings. Defaults to False.
            use_tfidf (bool): Whether to use TFIDF embeddings. Defaults to False.
        """
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []

        if use_bert and self.bert_tokenizer is None:
            raise AssertionError("BERT model not initialized")

        if use_bert:
            self.info["embeddings"] = f"BERT-{self.emb_model_name}"
            for _, row in self.trainDF.iterrows():
                embeddings = self._get_bert_embeddings(row["sentence"])
                self.X_train.append(torch.mean(embeddings, dim=0).numpy())
                self.y_train.append(row["relation"])
            for _, row in self.testDF.iterrows():
                embeddings = self._get_bert_embeddings(row["sentence"])
                self.X_test.append(torch.mean(embeddings, dim=0).numpy())
                self.y_test.append(row["relation"])
        elif use_tfidf:
            self.info["embeddings"] = "TF-IDF"
            # Remove special entity tags and get TF-IDF features
            try:
                self.trainDF["sentence"].replace("<e1>", "").replace("</e1>", "").replace("<e2>", "").replace("</e2>", "")
                self.testDF["sentence"].replace("<e1>", "").replace("</e1>", "").replace("<e2>", "").replace("</e2>", "")
            except:
                pass
            self.tfidf.fit(pd.concat([self.trainDF["sentence"], self.testDF["sentence"]]))
            self.X_train = self.tfidf.transform(self.trainDF["sentence"]).toarray()
            self.X_test = self.tfidf.transform(self.testDF["sentence"]).toarray()
            self.y_train = self.trainDF["relation"]
            self.y_test = self.testDF["relation"]
        else:
            self.info["embeddings"] = "Count Vectorizer (BoW)"
            # BoW
            self.bow.fit(pd.concat([self.trainDF["sentence"], self.testDF["sentence"]]))
            self.X_train = self.bow.transform(self.trainDF["sentence"]).toarray()
            self.X_test = self.bow.transform(self.testDF["sentence"]).toarray()
            self.y_train = self.trainDF["relation"]
            self.y_test = self.testDF["relation"]


class SVMs:
    """
    A class for training and evaluating Support Vector Machines (SVMs).
    """

    special_tokens = ["[E1]", "[/E1]", "[E2]", "[/E2]"]
    X_train, y_train, X_test, y_test = None, None, None, None
    bert_model, bert_tokenizer = None, None
    trainDF, testDF = None, None
    tfidf = tf_idf()
    bow = cv()
    emb_model_name = ""
    le = LabelEncoder()

    def __init__(self, trainDF, testDF, emb_model_name="prajjwal1/bert-tiny"):
        """
        Initialize the SVMs class.

        Args:
            trainDF (pd.DataFrame): The training data DataFrame.
            testDF (pd.DataFrame): The testing data DataFrame.
            emb_model_name (str): The name of the BERT model to load. Defaults to "prajjwal1/bert-tiny".
        """
        self.emb_model_name = emb_model_name
        self._load_bert(emb_model_name)
        self.trainDF = trainDF
        self.testDF = testDF

    def fit_and_score(self):
        """
        Fit and score the SVM model.

        Returns:
            dict: A dictionary containing the fitted model instance and metadata.
        """
        model = SVC(kernel='rbf')
        model.fit(self.X_train, self.y_train)
        print("\n")
        print(50 * "=" + "TRAINING" + 50 * "=")
        model.fit(self.X_train, self.y_train)

        # Calculate train accuracy
        train_accuracy = model.score(self.X_train, self.y_train)
        print(f"Train Accuracy: {train_accuracy:.2f}")

        print("\n")
        print(50 * "=" + "TESTING" + 50 * "=")
        # Calculate test accuracy
        test_accuracy = model.score(self.X_test, self.y_test)
        print(f"Test Accuracy: {test_accuracy:.2f}")
        print("\nTraining complete.")

        return {"model": model, "info": {"embeddings": "BERT-" + self.emb_model_name, "model": "SVC"}}

    def _load_bert(self, model_name):
        """
        Load the BERT model and tokenizer.

        Args:
            model_name (str): The name of the BERT model to load.
        """
        self.bert_tokenizer = BertTokenizerFast.from_pretrained(model_name)
        self.bert_model = BertModel.from_pretrained(model_name)
        self.bert_tokenizer.add_special_tokens({"additional_special_tokens": self.special_tokens})
        self.bert_model.resize_token_embeddings(len(self.bert_tokenizer))

    def _get_bert_embeddings(self, sentence):
        """
        Get BERT embeddings for a sentence.

        Args:
            sentence (str): The input sentence.

        Returns:
            torch.Tensor: The BERT embeddings for the sentence.
        """
        # Tokenize the sentence
        sentence_tokens = self.bert_tokenizer.tokenize(sentence)

        # Convert the sentence tokens to input IDs
        input_ids = self.bert_tokenizer.convert_tokens_to_ids(sentence_tokens)

        # Create the input tensor
        input_tensor = torch.tensor([input_ids])

        # Get BERT embeddings for the sentence
        with torch.no_grad():
            outputs = self.bert_model(input_tensor)
            sentence_embedding = outputs.last_hidden_state.squeeze(0)

        return sentence_embedding

    def prepare_dataset(self):
        """
        Prepare the dataset for model training and evaluation.
        """
        self.X_train = []
        self.y_train = []
        self.X_test = []
        self.y_test = []

        # Transform the trainDF['relation'] string classes into numbers (0->'class1', 1->'class2'...):
        self.le.fit(self.trainDF['relation'])
        self.y_train = self.le.transform(self.trainDF['relation'])
        self.y_test = self.le.transform(self.testDF['relation'])

        for _, row in self.trainDF.iterrows():
            embeddings = self._get_bert_embeddings(row["sentence"])
            self.X_train.append(torch.mean(embeddings, dim=0).numpy())
        for _, row in self.testDF.iterrows():
            embeddings = self._get_bert_embeddings(row["sentence"])
            self.X_test.append(torch.mean(embeddings, dim=0).numpy())
