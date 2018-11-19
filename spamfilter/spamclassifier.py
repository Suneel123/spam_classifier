from nltk.classify.util import apply_features
from nltk import NaiveBayesClassifier
from nltk.tokenize import regexp_tokenize
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import itertools
import pandas as pd
import os

from flask import current_app
import pickle, re
import collections


class SpamClassifier:

    def load_model(self, model_name):
        model_file = os.path.join(current_app.config['ML_MODEL_UPLOAD_FOLDER'], model_name+'.pk')
        model_word_features_file = os.path.join(current_app.config['ML_MODEL_UPLOAD_FOLDER'],model_name +'_word_features.pk')
        with open(model_file, 'rb') as mfp:
            self.classifier = pickle.load(mfp)
        with open(model_word_features_file, 'rb') as mwfp:
            self.word_features = pickle.load(mwfp)

    def extract_tokens(self, text, target):
        """returns array of tuples where each tuple is defined by (tokenized_text, label)
         parameters:
                text: array of texts
                target: array of target labels

        NOTE: consider only those words which have all alphabets and atleast 3 characters.
        """
        return [([word.lower() for word in regexp_tokenize(para, pattern=r"\w+|\$[\d\.]+|\S+")
                  if word.isalpha() and len(word) >= 3], label)
                for para, label in zip(text, target)]

    def get_features(self, corpus):
        """
        returns a Set of unique words in complete corpus.
        parameters:- corpus: tokenized corpus along with target labels

        Return Type is a set
        """
        tokenized_corpus, _ = zip(*corpus)
        return set(itertools.chain.from_iterable(tokenized_corpus))

    def extract_features(self, document):
        """
        maps each input text into feature vector
        parameters:- document: string

        Return type : A dictionary with keys being the train data set word features.
                      The values correspond to True or False
        """
        tokenized_document = regexp_tokenize(document, pattern=r"\w+|\$[\d\.]+|\S+")
        doc_vocab = set(tokenized_document)
        return {word: (word in doc_vocab) for word in self.word_features}

    def train(self, text, labels):
        """
        Returns trained model and set of unique words in training data
        """
        tokenized_corpus_labeled = self.extract_tokens(text, labels)
        self.word_features = self.get_features(tokenized_corpus_labeled)
        docs_features = [(self.extract_features(doc), label) for (doc, label) in zip(text, labels)]
        self.classifier = NaiveBayesClassifier.train(docs_features)
        return self.classifier, self.word_features

    def predict(self, text):
        """
        Returns prediction labels of given input text.
        """
        if isinstance(text, str):
            predictions = NaiveBayesClassifier.classify(self.classifier, self.extract_features(text))
        elif isinstance(text, list):
            predictions = [NaiveBayesClassifier.classify(self.classifier, self.extract_features(email))
                           for email in text]
        elif isinstance(text, dict):
            predictions = collections.OrderedDict({key: NaiveBayesClassifier.classify(self.classifier, self.extract_features(email))
                                                   for key, email in text.items()})
        return predictions


if __name__ == '__main__':

    print('Done')
