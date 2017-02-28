#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import pprint

import textimport
import textverarbeitung
import pickle

import logging as log
import argparse


def process_arguments():
    parser = argparse.ArgumentParser(description='Classification')
    parser.add_argument('--verbose', '-v', action='store_true', help='verbose flag')
    parser.add_argument('--debug', '-d', action='store_true', help='debug flag')
    parser.add_argument('--logfile', help="Write to this file instead of console")
    parser.add_argument('--action', required=True, choices=['learn', 'classify'])
    parser.add_argument('--learning-data', '-l', help='Input resp. output of learning data')
    parser.add_argument('data', nargs='+', help=
        'Data to process. Depending on the action (see below). For learning mode, the program expects a path to a file '
        'with tagged urls for learning. In the classification mode, you can either specify one or more URLs directly '
        'or a file with one URL per line for classification.'
    )

    return parser.parse_args()


################################################################################

# Länge=Wortanzahl ausgeben

def process_tagged_urls(learning_data):
    per_subject_word_freq = {}

    for subject in learning_data:
        for url in learning_data[subject]:
            log.info("Processing %s" % url)
            RAW_TEXT = textimport.load_text(url)

            if len(RAW_TEXT) > 0:
                freq = textverarbeitung.makeWordFrequencyDictionary(RAW_TEXT)
                log.debug("Word list: %s" % pprint.pformat(freq))

                per_subject_word_freq[subject] = textverarbeitung.appendWordFreqDictToExistingDict(
                    per_subject_word_freq.get(subject, dict()), freq)
            else:
                log.warning("ERROR: No text from %s" % url)

    textimport.write_textcache()
    return per_subject_word_freq


def write_learning_data_to_file(per_subject_word_freq, filename):
    wordfreq_dict_file = open(filename, mode="wb")
    pickle.dump(per_subject_word_freq, wordfreq_dict_file)


def load_learning_data_from_file(filename):
    wordfreq_dict_file = open(filename, mode="rb")
    return pickle.load(wordfreq_dict_file)


def setup_logging(args):
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


def do_learning(wordlist_fn, learning_data_files):
    learning_data = textimport.load_learning_data_from_file( learning_data_files )
    per_subject_word_freq = process_tagged_urls(learning_data)
    write_learning_data_to_file(per_subject_word_freq, wordlist_fn)

    for subject, wordfreq_dist in per_subject_word_freq.items():
        sorted_word_dist = textverarbeitung.buildSortedListFromDictionary(wordfreq_dist)

        log.info("Wortliste der Kategorie %s\n\n%s" % (subject, pprint.pformat(sorted_word_dist[:40])))


def do_classification(wordlist_fn, classification_data_paths):
    per_subject_word_freq = load_learning_data_from_file(wordlist_fn)

    for path in classification_data_paths:
        RAW_TEXT = textimport.load_text(path)
        if len(RAW_TEXT) > 0:
            per_subject_score = textverarbeitung.compareTextToLearningData(RAW_TEXT, per_subject_word_freq)
            log.info("Per subject score for %s:\n%s" % (path, pprint.pformat(per_subject_score)))
        else:
            log.warn("Cannot read text from %s" % path)


def main():
    args = process_arguments()

    setup_logging(args)

    wordlist_fn = args.learning_data if args.learning_data else "wordlists.obj"
    if args.action == "learn":
        do_learning(wordlist_fn, args.data)
    elif args.action == "classify":
        do_classification(wordlist_fn, args.data)

if __name__ == '__main__':
    main()
