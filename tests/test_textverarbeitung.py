#!/usr/bin/env python3

import unittest
import sys, os

# Modulsuchpfad erweitern
THIS_MODULES_PATH = os.path.dirname(__file__)
TEST_MODULES_PATH = os.path.join(THIS_MODULES_PATH, "..")

sys.path.append(TEST_MODULES_PATH)

# Importiere zu testendes Modul
from textverarbeitung import *

############################################################
# Unittest fuer textverarbeitung.py
############################################################

class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)


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
        INPUT_WORD_LIST = "My 32 pigs whistle.".split(" ")
        EXPECTED_OUTPUT = "My pigs whistle.".split(" ")

        self.assertEqual(cleanWordList(INPUT_WORD_LIST), 
            EXPECTED_OUTPUT
        )

    def test_if_words_are_split(self):
        INPUT_WORD_LIST = "My 32 pigs-whistle 32-times.".split(" ")
        EXPECTED_OUTPUT = "My pigs whistle times.".split(" ")        

        self.assertEqual(
            cleanWordList(INPUT_WORD_LIST), 
            EXPECTED_OUTPUT
        )



if __name__ == '__main__':
    unittest.main()
