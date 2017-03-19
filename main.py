#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import pprint

import textimport
import textverarbeitung
import pickle

import logging as log
import argparse


class LearningData:
    base = set()
    category_words = dict()
    data = []
    target = []


def processArguments():
    parser = argparse.ArgumentParser(description='Classification')
    parser.add_argument('--verbose', '-v', action='store_true', help='verbose flag')
    parser.add_argument('--debug', '-d', action='store_true', help='debug flag')
    parser.add_argument('--logfile', help="Write to this file instead of console")
    parser.add_argument('--action', required=True, choices=['learn', 'classify', 'test'])
    parser.add_argument('--learning-data', '-l', help='Input resp. output of learning data')
    parser.add_argument('data', nargs='+', help=
        'Data to process. Depending on the action (see below). For learning mode, the program expects a path to a file '
        'with tagged urls for learning. In the classification mode, you can either specify one or more URLs directly '
        'or a file containing text for classification. In test mode, a file like in the learning mode is used. But '
        'instead of learning the catecories, it''s checked, if the links in the file are classified correctly.'
    )

    return parser.parse_args()


################################################################################

# Länge=Wortanzahl ausgeben

def processTaggedUrlsWith(learning_data, merge_operation):
    per_subject_word_freq = {}

    for subject in learning_data:
        for url in learning_data[subject]:
            log.info("Processing %s" % url)
            RAW_TEXT = textimport.load_text(url)

            if len(RAW_TEXT) > 0:
                freq = textverarbeitung.makeWordFrequencyDictionary(RAW_TEXT)
                log.debug("Word list: %s" % pprint.pformat(freq))

                per_subject_word_freq[subject] = merge_operation(
                    per_subject_word_freq.get(subject, dict() ), url, freq )
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


def doLearning(wordlist_fn, learning_data_files):
    per_subject_urls = textimport.get_urls_per_subject_from_file(learning_data_files)
    per_subject_url_and_word_freq = processTaggedUrlsWith(per_subject_urls, addWordFreqPerUrl)
    per_subject_word_freq = dict()

    # build per subject wordfreq dictionary, keeping all the word lists in memory
    for subject, url_and_word_freq_dist in per_subject_url_and_word_freq.items():
        for url, wordfreq_dist in url_and_word_freq_dist.items():
            per_subject_word_freq[subject] = mergeWordFreqs(per_subject_word_freq.get(subject, dict()), "", wordfreq_dist)

    classification_base = textverarbeitung.buildClassificationSpaceBase(per_subject_word_freq)
    log.debug("Constructed the following base of length %i: %s" % (len(classification_base), pprint.pformat(classification_base)))

    learning_data = LearningData()
    learning_data.base = classification_base
    learning_data.category_words = {subject: classification_base & set(word_freqs.keys()) for subject, word_freqs in per_subject_word_freq.items()}

    log.debug("The categories have these words: %s" % pprint.pformat(learning_data.category_words))

    # now that we have a base, construct the vectors for each URL
    for subject, url_and_word_freq_dist in per_subject_url_and_word_freq.items():
        for url, wordfreq_dist in url_and_word_freq_dist.items():
            p = textverarbeitung.getClassificationVectorSpaceElement(classification_base, wordfreq_dist)
            learning_data.data.append(p)
            learning_data.target.append(subject)

            log.debug("Constructed vector %s" % pprint.pformat(p))

    writeLearningDataToFile(learning_data, wordlist_fn)



def doTesting(wordlist_fn, testing_data_files, classification_params):
    testing_data = textimport.get_urls_per_subject_from_file(testing_data_files)
    per_subject_url_and_word_freq = processTaggedUrlsWith(testing_data, addWordFreqPerUrl)
    learning_data = loadLearningDataFromFile(wordlist_fn)

    all_learned_subjects = set(learning_data.target)
    log.info("Starting to classify to these categories %s" % ", ".join(all_learned_subjects) )

    for subject, url_and_wordfreq_dist in per_subject_url_and_word_freq.items():
        for url, wordfreq_dist in url_and_wordfreq_dist.items():
            per_subject_score = textverarbeitung.compareWordFreqDictToLearningData(
                wordfreq_dist, learning_data, classification_params)
            classified_to = textverarbeitung.getWinningSubject(per_subject_score, classification_params)

            if classified_to == subject:
                log.info("%s correctly classified to %s" % (url, classified_to))
            elif classified_to == None:
                if subject not in all_learned_subjects:
                    log.info("%s correctly classified to other" % url)
                else:
                    log.warning("%s incorrectly classified to other, should be %s" % (url, subject))
                    log.warning("Scores were: %s" % pprint.pformat(per_subject_score))
            elif classified_to != subject:
                log.warning("%s incorrectly classified to %s." % (url, classified_to))
                log.warning("Scores were: %s" % pprint.pformat(per_subject_score))


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
        doLearning(learning_data_fn, args.data)
    else:
        classification_params = textverarbeitung.getClassificationStdParam()
        classification_params["min_difference_for_classication"] = 0.2

        if args.action == "classify":
            doClassification(learning_data_fn, args.data, classification_params)
        elif args.action == "test":
            doTesting(learning_data_fn, args.data, classification_params)

if __name__ == '__main__':
    main()
