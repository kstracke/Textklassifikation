# Alles rund ums Einlesen von Text aus verschiedenen Quellen

import re
import urllib.request

from bs4 import BeautifulSoup, SoupStrainer

################################################################################
## Text-Gewinnung
##
################################################################################

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


def load_text_from_url(url):
    req = urllib.request.Request(url, headers={'User-Agent': "Magic Browser"})
    con = urllib.request.urlopen(req)
    html = con.read()

    soup = BeautifulSoup(html, 'html.parser', parse_only=SoupStrainer('article'))

    texts = soup.find_all(text=True)

    visible_texts = filter(is_visible, texts)

    return "\n".join(visible_texts)


### TODO: Think about getting those links from ATOM feeds OR reading the while blog from ATOM feeds
### TODO: -->  https://pypi.python.org/pypi/feedparser
