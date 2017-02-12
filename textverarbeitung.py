# Verarbeiten des Eingelesenen Textes - Tokenisieren, Filtern

import re
import nltk

from collections import defaultdict # sum of dicts

from fractions import Fraction # rational numbers

from nltk.corpus import stopwords

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
        # 2. filtere nur a-z, A-Z,
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
    return dict(zip(wordlist, wordfreq))


# Worte nach Häufigkeit des Vorkommens sortieren
def sortFreqDict(freqdict):
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

    # Worte nach Haeufigkeit des Vorkommens sortieren
    return sortFreqDict(DICTIONARY)


def appendWordFreqDictToExistingDict(existing, to_append):
    for key, value in to_append.items():
        existing[key] = value + existing.get(key, Fraction(0,1))

    return existing
