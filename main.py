#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import pprint

import textimport
import textverarbeitung
import pickle

# schönere Ausgabe
pp = pprint.PrettyPrinter(indent=3)

################################################################################

# Länge=Wortanzahl ausgeben

learning_data = textimport.load_learning_data_from_file("Feedliste.txt")

log = open("log.txt",mode="w")

per_subject_word_freq = {}

for subject in learning_data:
    for url in learning_data[subject]:
        RAW_TEXT = textimport.load_text_from_url(url)

        log.write("***\n")
        log.write(url + "\n")
        log.write(RAW_TEXT + "\n")

        if len(RAW_TEXT) > 0:
            freq = textverarbeitung.makeWordFrequencyDictionary(RAW_TEXT)
            pprint.pprint(freq, log)

            per_subject_word_freq[subject] = textverarbeitung.appendWordFreqDictToExistingDict(
                per_subject_word_freq.get(subject, dict()), freq)
        else:
            log.write("ERROR: No text from this URL\n")

        log.flush()
        log.write("\n")

# gebe alle Wortfreq-Wörterbücher in Datei aus
wordfreq_dict_file = open("wordlists.obj", mode="wb")
pickle.dump(per_subject_word_freq, wordfreq_dict_file)

# pprint.pprint(per_subject_word_freq, log)

log.write("\n\nResult:\n")
for subject, wordfreq_dist in per_subject_word_freq.items():
    log.write("%s\n" % subject)

    sorted_word_dist = textverarbeitung.buildSortedListFromDictionary(wordfreq_dist)

    pprint.pprint(sorted_word_dist[:40], log)
