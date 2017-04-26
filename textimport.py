#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

# Alles rund ums Einlesen von Text aus verschiedenen Quellen

#import feedparser
import re
import urllib.request
import os
import pickle

from bs4 import BeautifulSoup

from breadability.readable import Article

from sortedcontainers import SortedSet, SortedDict

##########################################################################
# Text-Gewinnung
#
##########################################################################
TEXT_CACHE = {}

# filtere den sichtbaren Text: nur bestimmte Elemente werden ausgelesen!
def is_visible(element):
    element_text = str(element)
    # print("element_start:", element_text, "element_ende")
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    elif re.match('\s+', element_text):
        return False
    elif re.match('\s*<!--.*-->\s*', element_text):
        return False
    return True

def remove_non_ascii_chars(string):
    return string.encode("ascii", "ignore").decode("ascii")

def load_text_from_url(url):
    if not url in TEXT_CACHE:
        req = urllib.request.Request(url, headers={'User-Agent': "Magic Browser"})
        con = urllib.request.urlopen(req)
        html = con.read()

        # Nutzen von breadability, um den relevanten html-Code herauszufiltern
        filtered_html = Article(html, url=url).readable

        # in Text umwandeln
        soup = BeautifulSoup(filtered_html, 'html.parser')

        texts = soup.find_all(text=True)

        text_per_tag = filter(is_visible, texts)

        TEXT_CACHE[url] = remove_non_ascii_chars("\n".join(text_per_tag))

    return TEXT_CACHE[url]


def load_text_from_file(fn):
    if not os.path.exists(fn):
        return ""

    with open(fn) as file_handle:
        return file_handle.read()


def load_text(path):
    if re.match("http[s]?://", path):
        return load_text_from_url(path)
    else:
        return load_text_from_file(path)


def get_urls_per_subject_from_file(fn_list):
    urls_per_subject = SortedDict()

    # einfachen String in eine Liste von Strings umwandeln
    if isinstance(fn_list, str):
        fn_list = [ fn_list ]

    for fn in fn_list:
        if not os.path.exists(fn):
            continue

        with open(fn) as file_handle:
            for line in file_handle:
                if "|" in line:
                    key, url = [ x.strip() for x in line.split("|")]

                    if url == "": continue

                    if key in urls_per_subject:
                        urls_per_subject[key].add(url)
                    else:
                        urls_per_subject[key] = SortedSet({url})
                else:
                    continue

    return urls_per_subject


def read_textcache():
    if os.path.isfile(".textcache"):
        text_cache_file = open(".textcache", mode="rb")
        return pickle.load(text_cache_file)
    return {}

def write_textcache():
    text_cache_file = open(".textcache", mode="wb")
    pickle.dump(TEXT_CACHE, text_cache_file)


TEXT_CACHE = read_textcache()
