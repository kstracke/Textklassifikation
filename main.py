#!/usr/bin/env python3

import pprint

import textimport
import textverarbeitung

# schönere Ausgabe
pp = pprint.PrettyPrinter(indent=3)

################################################################################

# Länge=Wortanzahl ausgeben

learning_data = textimport.load_learning_data_from_file("Feedliste.txt")

log = open("log.txt",mode="w")


for subject in learning_data:
    for url in learning_data[subject]:
        RAW_TEXT = textimport.load_text_from_url(url)

        log.write("***\n")
        log.write(url + "\n")
        log.write(RAW_TEXT + "\n")

        freq = textverarbeitung.makeWordFrequencyDictionary(RAW_TEXT)
        pprint.pprint(freq, log)
        log.flush()
        log.write("\n")
