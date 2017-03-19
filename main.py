#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import pprint

import textimport
import textverarbeitung
import pickle

import logging as log
import argparse


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


def writeLearningDataToFile(per_subject_word_freq, filename):
    wordfreq_dict_file = open(filename, mode="wb")
    pickle.dump(per_subject_word_freq, wordfreq_dict_file)


def loadLearningDataFromFile(filename):
    wordfreq_dict_file = open(filename, mode="rb")
    return pickle.load(wordfreq_dict_file)


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
    learning_data = textimport.get_urls_per_subject_from_file(learning_data_files)
    per_subject_word_freq = processTaggedUrlsWith(learning_data, mergeWordFreqs)
    writeLearningDataToFile(per_subject_word_freq, wordlist_fn)

    for subject, wordfreq_dist in per_subject_word_freq.items():
        sorted_word_dist = textverarbeitung.buildSortedListFromDictionary(wordfreq_dist)

        log.info("Wortliste der Kategorie %s\n\n%s" % (subject, pprint.pformat(sorted_word_dist[:40])))


def doTesting(wordlist_fn, testing_data_files, classification_params):
    testing_data = textimport.get_urls_per_subject_from_file(testing_data_files)
    per_subject_url_and_word_freq = processTaggedUrlsWith(testing_data, addWordFreqPerUrl)
    learned_per_subject_word_freq = loadLearningDataFromFile(wordlist_fn)

    all_learned_subjects = learned_per_subject_word_freq.keys()
    log.info("Starting to classify to these categories %s" % ", ".join(all_learned_subjects) )

    for subject, url_and_wordfreq_dist in per_subject_url_and_word_freq.items():
        for url, wordfreq_dist in url_and_wordfreq_dist.items():
            per_subject_score = textverarbeitung.compareWordFreqDictToLearningData(
                wordfreq_dist, learned_per_subject_word_freq, classification_params)
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
    per_subject_word_freq = loadLearningDataFromFile(wordlist_fn)

    results = {}
    for path in classification_data_paths:
        RAW_TEXT = textimport.load_text(path)
        if len(RAW_TEXT) > 0:
            freq =  textverarbeitung.makeWordFrequencyDictionary(RAW_TEXT)
            per_subject_score = textverarbeitung.compareWordFreqDictToLearningData(
                freq, per_subject_word_freq, classification_params)
            results[path] = per_subject_score
        else:
            log.warning("Cannot read text from %s" % path)

    pprint.pprint(results)

def main():
    args = processArguments()

    setupLogging(args)

    wordlist_fn = args.learning_data if args.learning_data else "wordlists.obj"
    if args.action == "learn":
        doLearning(wordlist_fn, args.data)
    else:
        classification_params = textverarbeitung.getClassificationStdParam()
        classification_params["min_difference_for_classication"] = 0.2

        if args.action == "classify":
            doClassification(wordlist_fn, args.data, classification_params)
        elif args.action == "test":
            doTesting(wordlist_fn, args.data, classification_params)

if __name__ == '__main__':
    main()
