import nltk
import re
import urllib.request
from bs4  import  BeautifulSoup, SoupStrainer

from nltk.corpus import treebank

def is_visible(element):
    element_text = str(element)
    #print("element_start:", element_text, "element_ende")
    if element.parent.name in ['style', 'script', '[document]', 'head', 'title']:
        return False
    elif re.match('\s+', element_text):
        return False
    elif re.match('\s*<!--.*-->\s*', element_text):
        return False
    return True

with urllib.request.urlopen('http://www.aptgetupdate.de/2016/12/22/review-teufel-move-bt-%c2%b7-in-ear-bluetooth-kopfhoerer/') as response:
    html = response.read()
    soup = BeautifulSoup(html, 'html.parser', parse_only=SoupStrainer('article'))

    texts = soup.find_all(text=True)

    visible_texts = filter(is_visible, texts)

    all_the_text = "\n".join(visible_texts)

    #print(all_the_text)

    for token in nltk.pos_tag(nltk.word_tokenize(all_the_text, language="german"), tagset="universal"):
        print ("word: ", token)

