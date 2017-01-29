# Verarbeiten des Eingelesenen Textes - Tokenisieren, Filtern

import re
import nltk

from nltk.corpus import stopwords


# nltk.download('book')
# from nltk import word_tokenize,sent_tokenize
# from nltk.corpus import treebank
# from nltk.book import *


##################(##############################################################
# Liste bereinigen
def cleanWordList(in_list):
    out_list = []

    for word in in_list:
        # 1. text_lowercase = [x.lower() for x in text]
        # resp: element = element.lower()
                
        # 2. filtere nur a-z, A-Z,
        if not isValidWord(word):
            continue

        # 3. trenne auch am Bindestrich
        # verwendet die split-funktion eines strings; die liefert eine liste der teile
        for split_word in word.split("-"):
            if isValidWord(word):
                out_list.append(split_word)

    return out_list

def isValidWord(in_word):
    if not re.match('[a-zA-Z0-9]', in_word):
        return False

    # Hex-Zahlen rausfiltern
    if re.match('0[xX][0-9a-fA-F]+$', in_word):
        return False
    
    # Dezimalzahlen filtern (\d entspricht [0-9])
    if re.match('\d+[\.,]?\d*$', in_word):
        return False

    return True


# Worten die Haeufigkeit des Vorkommens zuordnen
def wordListToFreqDict(wordlist):
    # for token in nltk.pos_tag(nltk.word_tokenize(sorted_text, language="english"), tagset="universal"):
    #   pp.pprint("word: ", token)
    wordfreq = [wordlist.count(p) for p in wordlist]
    return dict(zip(wordlist, wordfreq))


# Worte nach Haeufigkeit des Vorkommens sortieren
def sortFreqDict(freqdict):
    aux = [(freqdict[key], key) for key in freqdict]
    aux.sort()
    aux.reverse()
    return aux

################################################################################

def getFilteredTokens(INPUT_TEXT):
    raw_tokens = nltk.word_tokenize(INPUT_TEXT, language="english")

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


    # NLTK-Stoppwoerter lokal in Liste schreiben
    stop_words = set(stopwords.words('english'))
    #pp.pprint(stopWords)
    # print("\n")

    return [w for w in sorted_text if w not in stop_words]

def makeWordFrequencyDictionary(INPUT_TEXT):
    # Text tokenisieren
    TOKENIZED_TEXT = getFilteredTokens(INPUT_TEXT)

    # Worten die Haeufigkeit ihres Vorkommens zuordnen
    DICTIONARY = wordListToFreqDict(TOKENIZED_TEXT)

    # Worte nach Haeufigkeit des Vorkommens sortieren
    return sortFreqDict(DICTIONARY)
