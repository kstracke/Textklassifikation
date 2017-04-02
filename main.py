#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import pprint

import textimport
import textverarbeitung
import pickle

import logging as log
import argparse
import sys

from numpy import array

from sortedcontainers import SortedSet, SortedDict

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB

class LearningData:
    def __init__(self):
        self.base = SortedSet()
        self.category_words = SortedDict()
        self.data = []
        self.target = []

class DummyScaler:
    def fit_transform(self, X):
        return X
    def transform(self, X):
        return X


def processArguments():
    classification_params = textverarbeitung.getClassificationStdParam()

    parser = argparse.ArgumentParser(description='Classification')
    parser.add_argument('--verbose', '-v', action='store_true', help='verbose flag')
    parser.add_argument('--debug', '-d', action='store_true', help='debug flag')
    parser.add_argument('--logfile', help="Write to this file instead of console")

    subparsers = parser.add_subparsers(dest="action")

    parser_learn = subparsers.add_parser('learn', help='learn from the given data')
    parser_learn.add_argument('--algorithm', default="svm", choices=["svm", "knn", "nb-multi"], help='classification algorithm')
    parser_learn.add_argument('--keep-shared-words', default=False, action='store_true',
                              help='keep words, which occur in every word list (default: remove)')
    parser_learn.add_argument('--learning-data', '-l', help='Write learning data to this file')
    parser_learn.add_argument('data', nargs='+', help=
        'Data to process. The program expects a path to a file with tagged urls for learning.')


    parser_test = subparsers.add_parser('test', help='test classification according to tagged data')
    parser_test.add_argument('--min-diff', type=float, default=classification_params['min_difference_for_classification'],
                             help='min_difference_for_classification')
    parser_test.add_argument('--other-cutoff', type=float, default=classification_params['other_cutoff'],
                             help='probability threshold to classify text as "other')
    parser_test.add_argument('--learning-data', '-l', help='Read learning data from this file')
    parser_test.add_argument('data', nargs='+', help=
        'Data to process. A file like in the learning mode is used. instead of learning the catecories, it''s checked, if the links in the file are classified correctly.')

    parser_test = subparsers.add_parser('classify', help='classify untagged data')
    parser_test.add_argument('--min-diff', type=float, default=classification_params['min_difference_for_classification'],
                             help='min_difference_for_classification')
    parser_test.add_argument('--other-cutoff', type=float, default=classification_params['other_cutoff'],
                             help='probability threshold to classify text as "other')
    parser_test.add_argument('--learning-data', '-l', help='Read learning data from this file')
    parser_test.add_argument('data', nargs='+', help=
    'Data to process. A file containing the text to classify  or an URL is expected')

    return parser.parse_args()


################################################################################

# Länge=Wortanzahl ausgeben

def processTaggedUrlsWith(learning_data, merge_operation):
    per_subject_word_freq = SortedDict()

    for subject in learning_data:
        for url in learning_data[subject]:
            log.info("Processing %s" % url)
            RAW_TEXT = textimport.load_text(url)

            if len(RAW_TEXT) > 0:
                freq = textverarbeitung.makeWordFrequencyDictionary(RAW_TEXT)
                log.debug("Word list: %s" % pprint.pformat(freq))

                per_subject_word_freq[subject] = merge_operation(
                    per_subject_word_freq.get(subject, SortedDict()), url, freq )
            else:
                log.warning("ERROR: No text from %s" % url)

    textimport.write_textcache()
    return per_subject_word_freq


def mergeWordFreqs(current_value, url, word_freq):
    return textverarbeitung.appendWordFreqDictToExistingDict(current_value, word_freq)


def addWordFreqPerUrl(current_value, url, word_freq):
    current_value[url] = word_freq
    return current_value


def writeLearningDataToFile(learning_data, filename):
    learning_data_file = open(filename, mode="wb")
    pickle.dump(learning_data, learning_data_file)


def loadLearningDataFromFile(filename):
    learning_data_file = open(filename, mode="rb")
    return pickle.load(learning_data_file)


def setupLogging(args):
    if args.verbose:
        verbosity = log.INFO
    else:
        verbosity = log.WARNING

    if args.debug:
        verbosity = log.DEBUG

    if args.logfile:
        log.basicConfig(filename=args.logfile, level=verbosity)
    else:
        log.basicConfig(level=verbosity)


def doLearning(wordlist_fn, learning_data_files, classification_params):
    per_subject_urls = textimport.get_urls_per_subject_from_file(learning_data_files)
    per_subject_url_and_word_freq = processTaggedUrlsWith(per_subject_urls, addWordFreqPerUrl)
    per_subject_word_freq = SortedDict()

    # build per subject wordfreq dictionary, keeping all the word lists in memory
    for subject, url_and_word_freq_dist in per_subject_url_and_word_freq.items():
        for url, wordfreq_dist in url_and_word_freq_dist.items():
            per_subject_word_freq[subject] = mergeWordFreqs(per_subject_word_freq.get(subject, SortedDict()), "", wordfreq_dist)

    classification_base = textverarbeitung.buildClassificationSpaceBase(per_subject_word_freq, classification_params)
    log.debug("Constructed the following base of length %i: %s" % (len(classification_base), pprint.pformat(classification_base)))

    learning_data = LearningData()
    learning_data.base = classification_base
    learning_data.category_words = SortedDict({subject: classification_base & set(word_freqs.keys()) for subject, word_freqs in per_subject_word_freq.items()})

    log.debug("The categories have these words: %s" % pprint.pformat(learning_data.category_words))

    # now that we have a base, construct the vectors for each URL
    for subject, url_and_word_freq_dist in per_subject_url_and_word_freq.items():
        for url, wordfreq_dist in url_and_word_freq_dist.items():
            p = textverarbeitung.getClassificationVectorSpaceElement(classification_base, wordfreq_dist)
            learning_data.data.append(p)
            learning_data.target.append(subject)

            log.debug("Constructed vector %s" % pprint.pformat(p))

    # Split the dataset in two equal parts
    if classification_params["algorithm"] == "nb-multi":
        scaler = StandardScaler(with_mean=False)
    else:
        scaler = StandardScaler()
    learning_data.data = scaler.fit_transform(learning_data.data)
    learning_data.scaler = scaler

    log.debug(pprint.pformat(scaler))

    X_train, X_test, y_train, y_test = train_test_split(
        array(learning_data.data), learning_data.target, test_size=0.2, random_state=0
    )

    log.debug("Data used for training:")
    log.debug(pprint.pformat(list(zip(y_train, X_train))))
    log.debug("Data used for testing:")
    log.debug(pprint.pformat(list(zip(y_test, X_test))))

    knear_tuned_parameters = SortedDict({
        'n_neighbors': [3, 4, 5, 6],
        'weights': ['uniform', 'distance']
    })

    svm_tuned_parameters = SortedDict({
        'kernel': ['linear', 'poly'],
        'C': [0.1, 1, 10, 100, 1000]
    })

    naive_bayes_multinominal_tuned_parameters = SortedDict({
        'alpha': [0.0, 0.5, 1.0, 2.0],
        'fit_prior': [True, False]
    })

    scores = ['precision', 'recall']

    for score in scores:
        print("# Tuning hyper-parameters for %s" % score)
        print()

        if classification_params["algorithm"] == "knn":
            clf = GridSearchCV(KNeighborsClassifier(), knear_tuned_parameters, cv=5)
        elif classification_params["algorithm"] == "svm":
            clf = GridSearchCV(SVC(probability=True, tol=1e-5), svm_tuned_parameters, cv=5)
        elif classification_params["algorithm"] == "nb-multi":
            clf = GridSearchCV(MultinomialNB(), naive_bayes_multinominal_tuned_parameters, cv=5)
        else:
            log.error("Unsupported algorithm " % classification_params["algorithm"])
            sys.exit(1)
        clf.fit(X_train, y_train)

        print("Best parameters set found on development set:")
        print()
        print(clf.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = clf.cv_results_['mean_test_score']
        stds = clf.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, clf.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"
                  % (mean, std * 2, params))
        print()

        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, clf.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()
        learning_data.classifier = clf.best_estimator_

    writeLearningDataToFile(learning_data, wordlist_fn)



def doTesting(wordlist_fn, testing_data_files, classification_params):
    testing_data = textimport.get_urls_per_subject_from_file(testing_data_files)
    per_subject_url_and_word_freq = processTaggedUrlsWith(testing_data, addWordFreqPerUrl)
    learning_data = loadLearningDataFromFile(wordlist_fn)

    learning_data.all_learned_subjects = SortedSet(learning_data.target)

    log.info("Starting to classify to these categories %s" % ", ".join(learning_data.all_learned_subjects) )
    print(learning_data.base)
    print(learning_data.classifier)

    all_counter = 0
    correct_counter = 0

    # table title
    result_tab = ["&\t".join([x[:6] for x in learning_data.all_learned_subjects]) + "&\testim.&\tshould \\\\"]

    for subject, url_and_wordfreq_dist in per_subject_url_and_word_freq.items():
        for url, wordfreq_dist in url_and_wordfreq_dist.items():
            all_counter = all_counter + 1
            per_subject_score = textverarbeitung.compareWordFreqDictToLearningData(
                wordfreq_dist, learning_data, classification_params)
            classified_to = textverarbeitung.getWinningSubject(per_subject_score, classification_params)

            row = [("%.3f" % x) for x in per_subject_score.values()]
            if classified_to == subject:
                log.info("%s correctly classified to %s" % (url, classified_to))
                row.append(classified_to[:6])
                row.append(subject[:6])
                correct_counter = correct_counter + 1
            elif classified_to is None:
                row.append("other")
                if subject not in learning_data.all_learned_subjects:
                    log.info("%s correctly classified to other" % url)
                    correct_counter = correct_counter + 1
                    row.append("other")
                else:
                    log.warning("%s incorrectly classified to other, should be %s" % (url, subject))
                    log.warning("Scores were: %s" % pprint.pformat(per_subject_score))
                    row.append(subject[:6])
            elif classified_to != subject:
                log.warning("%s incorrectly classified to %s." % (url, classified_to))
                log.warning("Scores were: %s" % pprint.pformat(per_subject_score))
                row.append(classified_to[:6])
                row.append("other" if subject not in learning_data.all_learned_subjects else subject[:6])

            result_tab.append("&\t".join(row))
    log.info("Tabular result:\n" + " \\\\\n".join(result_tab) + " \\\\\n")
    log.info("Correct classified: %f %%" % (correct_counter*100.0 / all_counter))


def doClassification(wordlist_fn, classification_data_paths, classification_params):
    learning_data = loadLearningDataFromFile(wordlist_fn)

    results = {}
    for path in classification_data_paths:
        RAW_TEXT = textimport.load_text(path)
        if len(RAW_TEXT) > 0:
            freq =  textverarbeitung.makeWordFrequencyDictionary(RAW_TEXT)
            per_subject_score = textverarbeitung.compareWordFreqDictToLearningData(
                freq, learning_data, classification_params)
            results[path] = per_subject_score
        else:
            log.warning("Cannot read text from %s" % path)

    pprint.pprint(results)

def main():
    args = processArguments()

    setupLogging(args)

    learning_data_fn = args.learning_data if args.learning_data else "learningdata.obj"
    if args.action == "learn":
        classification_params = textverarbeitung.getClassificationStdParam()
        classification_params["algorithm"] = args.algorithm
        classification_params["remove_shared_words"] = False if args.keep_shared_words else True

        doLearning(learning_data_fn, args.data, classification_params)
    else:
        classification_params = textverarbeitung.getClassificationStdParam()
        classification_params["min_difference_for_classification"] = args.min_diff
        classification_params["other_cutoff"] = args.other_cutoff

        if args.action == "classify":
            doClassification(learning_data_fn, args.data, classification_params)
        elif args.action == "test":
            doTesting(learning_data_fn, args.data, classification_params)

if __name__ == '__main__':
    main()
