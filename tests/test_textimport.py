#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import unittest
import sys, os

# Modulsuchpfad erweitern
THIS_MODULES_PATH = os.path.dirname(__file__)
TEST_MODULES_PATH = os.path.join(THIS_MODULES_PATH, "..")

sys.path.append(TEST_MODULES_PATH)

# Importiere zu testendes Modul
from textimport import *

############################################################
# Unittest fuer textimport.py
############################################################

class TestLoadLearningData(unittest.TestCase):

    def test_returns_empy_dict_on_not_existing_file(self):
        NOT_EXISTING=os.path.join("Path", "To", "Most", "Likely", "not", "xisting", "file")
        self.assertEqual( load_learning_data_from_file(NOT_EXISTING), {})

    def test_can_load_example_file(self):
        INPUT_FILE=os.path.join("tests", "data", "test_textimport.txt")
        CONTEXT={ 
            "cooking" : ["http://chefkoch.de", "http://kochen.de"],
            "cars" :   [ "http://auto-motor-sport.de"]
        }
        self.assertEqual(load_learning_data_from_file(INPUT_FILE), CONTEXT)

if __name__ == '__main__':
    unittest.main()
