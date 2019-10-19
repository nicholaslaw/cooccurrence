from sklearn import base
from sklearn.feature_extraction.text import VectorizerMixin
import numpy as np

class CooccurrenceVectorizer(base.BaseEstimator, VectorizerMixin):
    def __init__(self, max_features=None, context_window=2):
        """
        max_features: int
            take top features based on total occurrence counts
        context_window: int
            size of context window
        """
        if max_features is not None:
            if not isinstance(max_features, int):
                raise TypeError("max_features must be an integer")
        if not isinstance(context_window, int):
            raise TypeError("max_features must be an integer")
        self.max_features = max_features
        self.context_window = context_window
        self.vocab = None
        self.cooc_matrix = None

    def fit(self, texts):
        """
        texts: list
            list of lists, each list contains tokens

        Builds/Updates vocabulary dictionary and cooccurrence matrix
        """
        if not (isinstance(texts, np.ndarray) or isinstance(texts, list)):
            raise TypeError("texts must be a list or numpy array")
        if self.vocab is None:
            self.build_vocab(texts)
        else:
            self.update_vocab(texts)
        self.build_matrix(texts)

        return self

    def transform(self, texts):
        """
        texts: list
            list of lists, each list contains tokens

        replace each token with their respective vector for ever data sample, returns a numpy matrix
        """
        result = []
        num_vocab = len(self.vocab)
        for text in texts:
            temp = []
            for token in text:
                idx = self.vocab.get(token, None)
                if idx is None:
                    word_vec = np.zeros((1, num_vocab))
                else:
                    word_vec = self.cooc_matrix[idx, :]
                temp.append(word_vec)
            result.append(np.array(temp))
        return result
    
    def fit_transform(self, texts):
        self.fit(texts)
        return self.transform(texts)

    def build_vocab(self, texts):
        """
        texts: list
            list of lists, each list contains tokens
        
        Builds vocabulary, only used when calling .fit for the first time
        """
        all_vocab = set([])
        for text in texts:
            all_vocab = all_vocab | set(text)
        self.vocab = {vocab: idx for idx, vocab in enumerate(all_vocab)}
        num_vocab = len(self.vocab)
        self.cooc_matrix = np.zeros((num_vocab, num_vocab))

    def update_vocab(self, texts):
        """
        word: str
            a token

        update vocabulary dictionary and append columns and rows to current occurrence, only called if .fit method is not called the first time
        """
        if self.vocab is None:
            raise Exception("Vocabulary Dictionary is empty, must call .fit method first")
        new_vocab = set([])
        for text in texts:
            new_vocab = new_vocab | set(text)
        current_vocab = set(list(self.vocab.keys()))
        current_num = len(current_vocab) # number of words in current vocabulary
        new_vocab -= current_vocab
        for idx, word in enumerate(new_vocab):
            self.vocab[word] = current_num + idx
            self.cooc_matrix = np.hstack((self.cooc_matrix, np.zeros((current_num + idx, 1)))) # append new column to matrix
            self.cooc_matrix = np.vstack((self.cooc_matrix, np.zeros((1, current_num + idx + 1)))) # append new row to matrix

    def build_matrix(self, texts):
        """
        texts: list
            list of lists, each list contains tokens

        Builds occurrence matrix, only called when calling .fit method for the first time
        """
        for text in texts:
            text_len = len(text)
            for idx, token in enumerate(text):
                start = idx - self.context_window
                end = idx + self.context_window
                for i in range(max(start, 0), min(end+1, text_len)):
                    if i != idx:
                        self.cooc_matrix[self.vocab[token], self.vocab[text[i]]] += 1
    
    def check_vocab(self, word):
        """
        word: str
            a token

        check whether word is in current vocabulary dictionary, True if it is
        """
        if not isinstance(word, str):
            raise TypeError("word must be a string")
        if self.vocab is None:
            raise Exception("Vocabulary Dictionary is empty, must call .fit method first")
        return self.vocab.get(word, None) is not None