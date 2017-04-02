
import logging as log
import numpy
import pprint

from numpy import array, zeros
from sklearn.base import BaseEstimator
from sortedcontainers import SortedSet, SortedDict

class DummyScaler:
    def fit_transform(self, X):
        return X
    def transform(self, X):
        return X

class SelfmadeNaive(BaseEstimator):
    def __init__(self):
        self.per_category_base = SortedDict({})
        self.categories = SortedSet()

    def fit(self, X, tags):
        for x, category in zip(X, tags):
            self.per_category_base[category] = x + self.per_category_base.get(category, zeros(len(x)))
            self.categories.add(category)

    def score(self, Xs, tags):
        result = [prediction == tag for prediction, tag in zip(self.predict(Xs), tags)]
        correct_ones = sum(result)
        return correct_ones / len(result)

    def predict_proba(self, Xs):
        probas = []
        for X in Xs:
            log.warning("Predicted probabilities input: %s" % pprint.pformat(X))
            result = []
            for category, category_vec in self.per_category_base.items():
                score = numpy.sum( (category_vec > 0) * X)

                log.debug("How it compares to %s: %f" % (category, score))

                #score = N_WORDS_TOT * sum_of_probabilities / FIRST_N_WORDS
                result.append(score)
            probas.append(array(result))
        log.warning("Predicted probabilities: %s" % pprint.pformat(probas))
        return probas

    def predict(self, Xs):
        return [self.categories[numpy.argmax(scores)] for scores in self.predict_proba(Xs)]

