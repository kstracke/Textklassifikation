#!/usr/bin/env python3

import pprint

import textimport
import textverarbeitung

# schönere Ausgabe
pp = pprint.PrettyPrinter(indent=3)

################################################################################

# Länge=Wortanzahl ausgeben

RAW_TEXT = textimport.load_text_from_url("http://os.phil-opp.com/multiboot-kernel.html")

print(len(RAW_TEXT))

pp.pprint(textverarbeitung.makeWordFrequencyDictionary(RAW_TEXT))
