# Alles rund ums Einlesen von Text aus verschiedenen Quellen

#import feedparser
import re
import urllib.request
import os

from bs4 import BeautifulSoup, SoupStrainer

from breadability.readable import Article

##########################################################################
# Text-Gewinnung
#
##########################################################################

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
    req = urllib.request.Request(url, headers={'User-Agent': "Magic Browser"})
    con = urllib.request.urlopen(req)
    html = con.read()

    # use breadability to extract the main html
    filtered_html = Article(html, url=url).readable

    # convert to text
    soup = BeautifulSoup(filtered_html, 'html.parser')

    texts = soup.find_all(text=True)

    text_per_tag = filter(is_visible, texts)

    return remove_non_ascii_chars("\n".join(text_per_tag))

def load_learning_data_from_file(fn):
    urls_per_subject = {}
    if not os.path.exists(fn):
        return urls_per_subject

    with open(fn) as file_handle:
        for line in file_handle:
            if "|" in line:
                key, url = [ x.strip() for x in line.split("|")]

                if url == "": continue

                if key in urls_per_subject:
                    urls_per_subject[key].append(url)
                else:
                    urls_per_subject[key] = [ url ]
            else:
                continue

    return urls_per_subject

# TODO: Think about getting those links from ATOM feeds OR reading the while blog from ATOM feeds
# TODO: -->  https://pypi.python.org/pypi/feedparser
### d= feedparser.parse('URL')
