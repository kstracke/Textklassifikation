#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Verarbeiten des Eingelesenen Textes - Tokenisieren, Filtern

import re
import nltk
import logging as log
import pprint
import numpy
import classification

from functools import  reduce

from collections import defaultdict # sum of dicts

from fractions import Fraction # rational numbers

from nltk.corpus import stopwords
from sortedcontainers import SortedSet, SortedDict

# NLTK-Stoppwörter global in Set importieren
STOP_WORDS = set(stopwords.words('english'))

from nltk.stem.snowball import SnowballStemmer

STEMMER = SnowballStemmer("english", ignore_stopwords=True)

##################(##############################################################
# Liste bereinigen
def cleanWordList(in_list):
    out_list = []

    # 1. text_lowercase
    in_list = [x.lower() for x in in_list]

    for word in in_list:
        # 2. filtere nur a-z, A-Z, 0-9
        if not isValidWord(word):
            continue

        # 3. trenne auch am Bindestrich
        # verwendet die split-funktion eines strings; die liefert eine Liste der Teile
        for split_word in word.split("-"):
            if isValidWord(split_word):
                #Stemiing = Worte werden auf den Wortstamm zurückgeführt
                stem_word = STEMMER.stem(split_word)
                out_list.append(stem_word)

    return out_list

def isValidWord(in_word):
    if not re.match('[a-zA-Z0-9]', in_word):
        return False

    # Hex-Zahlen rausfiltern
    if re.match('0[xX][0-9a-fA-F]+$', in_word):
        return False
    
    # Dezimalzahlen filtern (\d entspricht [0-9])
    if re.match('\d+[.,]?\d*$', in_word):
        return False

    return True


# Worten die Häufigkeit des Vorkommens zuordnen
def wordListToFreqDict(wordlist, scale=1):
    # for token in nltk.pos_tag(nltk.word_tokenize(sorted_text, language="english"), tagset="universal"):
    #   pp.pprint("word: ", token)
    wordfreq = [ Fraction(wordlist.count(p), scale) for p in wordlist]
    return SortedDict(zip(wordlist, wordfreq))


# Worte nach Häufigkeit des Vorkommens sortieren
def buildSortedListFromDictionary(freqdict):
    aux = [(freqdict[key], key) for key in freqdict]
    aux.sort()
    aux.reverse()
    return aux

################################################################################


def getFilteredTokens(INPUT_TEXT):
    raw_tokens = list(nltk.word_tokenize(INPUT_TEXT, language="english"))
    number_of_tokens = len(raw_tokens)

    ## (POS-Tokenisierung
    #posTokenizedText = nltk.pos_tag(nltk.word_tokenizeRAW_TEXT, language="english"), tagset="universal")
    #sortedTokens = sorted(posTokenizedText, key=lambda token: token[0])
    #pp.pprint(sortedTokens)
    ##
    # Liste bereinigen, siehe oben
    cleaned_text = cleanWordList(raw_tokens)
    #pp.pprint(cleanedText)

    # Liste alphabetisch sortieren
    sorted_text = (sorted(cleaned_text))
    #pp.pprint(sortedText)


    return ( [w for w in sorted_text if w not in STOP_WORDS], number_of_tokens )


#Wörterbuch mit Wortanzahl als key, Wörtern als value erstellen

def makeWordFrequencyDictionary(INPUT_TEXT):
    # Text tokenisieren
    (TOKENIZED_TEXT, INPUT_WORD_COUNT) = getFilteredTokens(INPUT_TEXT)

    DICTIONARY = wordListToFreqDict(TOKENIZED_TEXT, INPUT_WORD_COUNT)

    return DICTIONARY


def appendWordFreqDictToExistingDict(existing, to_append):
    for key, value in to_append.items():
        existing[key] = value + existing.get(key, Fraction(0,1))

    return existing


def compareWordFreqDictToLearningData(freq, learning_data, params):
    FIRST_N_WORDS=40
    log.debug("Words: %s" % pprint.pformat(freq))


    p = learning_data.scaler.transform(
        getClassificationVectorSpaceElement(learning_data.base, freq).reshape(1,-1)
    )
    if isinstance(learning_data.classifier, classification.SelfmadeNaive):
        # For our selfmade classifier, we need to scale the vectors differently
        N_WORDS_TOT = len(freq.keys())
        WORDS_PER_BASE = len(learning_data.base) / len(learning_data.all_learned_subjects)
        p *= N_WORDS_TOT / WORDS_PER_BASE

    res = learning_data.classifier.predict_proba(p.reshape(1,-1))
    #res = learning_data.classifier.decision_function(p.reshape(1,-1))

    result = SortedDict({subject: score for subject, score in zip(learning_data.all_learned_subjects, res[0])})
    return result

def getClassificationStdParam():
    param = {}
    param["min_difference_for_classification"] = 0.2
    param["other_cutoff"] = 0.6
    param["algorithm"] = "svm"
    param["category_base_length"] = 40  # aka first_n_words
    param["remove_shared_words"] = True

    return param


def getWinningSubject(per_subject_score, classification_params):
    if len(per_subject_score) == 0:
        return None

    max_score = max(per_subject_score.values())
    if max_score < classification_params["other_cutoff"]:
        return None

    max_score_scale = 1.0 / max(per_subject_score.values())
    classification_thres = 1.0 - classification_params["min_difference_for_classification"]

    failed_subjects = set([subject for subject, score in per_subject_score.items()
                           if score*max_score_scale < classification_thres])

    winning_subjects = set(per_subject_score.keys()) - failed_subjects

    if len(winning_subjects) > 1:
        return None
    else:
        return winning_subjects.pop()

def buildClassificationSpaceBase(per_subject_wordfreq_dict, classification_params):
    first_n_words = classification_params["category_base_length"]

    result = SortedSet()

    all_the_words = [SortedSet(word_freqs.keys()) for word_freqs in per_subject_wordfreq_dict.values()]

    if classification_params["remove_shared_words"]:
        shared_words = reduce(lambda x, y: x & y, all_the_words) if len(all_the_words) > 1 else SortedSet()
    else:
        shared_words = SortedSet()

    log.info("Those words exist in every category: %s" % pprint.pformat(shared_words))

    for category, wordfreq_dist in per_subject_wordfreq_dict.items():
        log.info("Processing category %s" % category)
        words = SortedSet([x[1] for x in buildSortedListFromDictionary(wordfreq_dist) if x[1] not in shared_words][:first_n_words])
        intersection = words & result

        if len(intersection) > 0:
            log.warning("The base of category %s seems not to be unique! Those base elements already exist: %s" %
                        (category, pprint.pformat(intersection)))

        result |= words

    return result

def getClassificationVectorSpaceElement(base, word_freq_dict):
    if len(base) > 0:
        return numpy.array([float(word_freq_dict.get(word, 0.0)) for word in base])
    else:
        return None
