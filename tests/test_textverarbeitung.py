#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import unittest
import sys, os
import numpy

# Modulsuchpfad erweitern
THIS_MODULES_PATH = os.path.dirname(__file__)
TEST_MODULES_PATH = os.path.join(THIS_MODULES_PATH, "..")

sys.path.append(TEST_MODULES_PATH)

# Importiere zu testendes Modul
from textverarbeitung import *

############################################################
# Unittest fuer textverarbeitung.py
############################################################

class TestIsValidWord(unittest.TestCase):

    def test_if_words_are_valid(self):

        #bricht nach dem ersten Fehler ab
        ###TODO: sicherstellen, dass alle Tests durchlaufen

        self.assertTrue(isValidWord("bit"))

        self.assertTrue(isValidWord("32bit"))

        self.assertFalse(isValidWord("32"))

        self.assertTrue(isValidWord("32-times"))

        self.assertFalse(isValidWord("0xbadc0ffee"))

        self.assertFalse(isValidWord("325,24"))

        self.assertFalse(isValidWord("345.6"))



class TestCleanWordList(unittest.TestCase):

    def test_invalid_words_are_removed(self):
        INPUT_WORD_LIST = "My 32 pigs whistle".split(" ")
        EXPECTED_OUTPUT = "my pig whistl".split(" ")

        self.assertEqual(cleanWordList(INPUT_WORD_LIST), 
            EXPECTED_OUTPUT
        )

    def test_if_words_are_split(self):
        INPUT_WORD_LIST = "My 32 pigs-whistle 32-times".split(" ")
        EXPECTED_OUTPUT = "my pig whistl time".split(" ")

        self.assertEqual(
            cleanWordList(INPUT_WORD_LIST), 
            EXPECTED_OUTPUT
        )


class TestMergeDictionary(unittest.TestCase):

    def test_identity(self):
        INPUT = { "hallo":1, "welt":1 }

        self.assertEqual(appendWordFreqDictToExistingDict(INPUT, {}), INPUT)

    def test_expand(self):
        INPUT = { "hallo":1, "welt":1 }
        OLD = {}

        self.assertEqual(appendWordFreqDictToExistingDict(OLD, INPUT), INPUT)

    def test_add_content(self):
        INPUT = { "hallo":1, "welt":1 }
        OLD = { "wie" : 2, "gehts": 1}
        RESULT = { "hallo":1, "welt":1, "wie" : 2, "gehts": 1}

        self.assertEqual(appendWordFreqDictToExistingDict(OLD, INPUT), RESULT)


class TestGetWinningSubject(unittest.TestCase):

    def test_empty_result(self):
        INPUT = {}

        self.assertEqual(getWinningSubject(INPUT, getClassificationStdParam()), None)

    def test_get_correct_winner(self):
        INPUT = { "stupid_option" : 1.0, "good_alternative" : 0.5, "better_alternative" : 0.1}

        self.assertEqual(getWinningSubject(INPUT, getClassificationStdParam()), "stupid_option")

    def test_get_scaling_independency(self):
        INPUT_A = { "stupid_option" : 1.0, "good_alternative" : 0.5, "better_alternative" : 0.1}
        INPUT_B = { "stupid_option" : 9.0, "good_alternative" : 4.5, "better_alternative" : 0.9}

        param = getClassificationStdParam()
        self.assertEqual(getWinningSubject(INPUT_A, param), getWinningSubject(INPUT_B, param))

    def test_its_without_an_alternative(self):
        INPUT = { "alternativlos" : 1  }

        self.assertEqual(getWinningSubject(INPUT, getClassificationStdParam()), "alternativlos")

    def test_too_close_to_be_sure(self):
        INPUT = { "nuke_the_planet" : 0.99, "live_peacefully" : 1.0}

        self.assertEqual(getWinningSubject(INPUT, getClassificationStdParam()), None)


class TestBuildClassificationVectorSpaceBase(unittest.TestCase):

    def test_empty(self):
        INPUT = {'subject1': {}}
        OUTPUT = set()
        PARAM = getClassificationStdParam()

        self.assertEqual(buildClassificationSpaceBase(INPUT, PARAM), OUTPUT)


    def test_overlap(self):
        INPUT = {'subject1': {'word1': 0.1, 'word2': 0.2},
                 'subject2': {'word2': 0.01, 'word3': 0.5},
                 'subject3': {'word4': 1.0, 'word1': 1.0}
                }
        OUTPUT = {'word1', 'word2', 'word3', 'word4'}
        PARAM = getClassificationStdParam()

        self.assertEqual(buildClassificationSpaceBase(INPUT, PARAM), OUTPUT)


    def test_remove_shared(self):
        INPUT = {'subject1': {'word1': 0.1, 'word2': 0.2},
                 'subject2': {'word2': 0.01, 'word3': 0.5},
                 'subject3': {'word2': 0.5, 'word4': 1.0}
                 }
        OUTPUT = {'word1', 'word3', 'word4'}
        PARAM = getClassificationStdParam()

        self.assertEqual(buildClassificationSpaceBase(INPUT, PARAM), OUTPUT)

    def test_length_limited(self):
        MAX_N = 40
        INPUT = {'subject1': {'word' + str(i): 1.0/(i+1) for i in range(MAX_N+1)}}
        OUTPUT = {'word' + str(i) for i in range(MAX_N)}
        PARAM = getClassificationStdParam()

        self.assertEqual(buildClassificationSpaceBase(INPUT, PARAM), OUTPUT)


class TestGetClassificationVectorSpaceElement(unittest.TestCase):

    def test_null(self):
        BASE = set()
        INPUT = {'word1': 0.1, 'word2': 0.2}
        OUTPUT = None

        self.assertTrue( (getClassificationVectorSpaceElement(BASE, INPUT) is OUTPUT))

    def test_base_smaller_than_input(self):
        BASE = {'word1', 'word2'}
        INPUT = {'word1': 0.1, 'word2': 0.2, 'word3': 1.0}
        OUTPUT = numpy.array([0.1, 0.2])

        self.assertTrue((getClassificationVectorSpaceElement(BASE, INPUT) == OUTPUT).all() )

if __name__ == '__main__':
    unittest.main()
