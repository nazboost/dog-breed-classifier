""" Search target keyword. """

import sys
import warnings

import requests
import wikipedia

warnings.filterwarnings('ignore')


def search(keyword):
    """
    Search keyword's summary of Wikipedia.

    Args:
        keyword (str):
            Seaarch target
    """

    keyword = keyword.replace('_', ' ')
    keyword = dog_breed_check(keyword)
    keyword = translate_en2ja(keyword)

    wikipedia.set_lang('jp')

    # If error occured owing to summary has some result, search third one.
    # FIXME: If input 'corgi', sometimes output wrong answer.
    try:
        summary = wikipedia.summary(keyword)
    except BaseException:
        summary = wikipedia.summary(wikipedia.search(keyword)[2])

    return summary


def dog_breed_check(keyword):
    """ Convert word for search on wikipedia. """
    if keyword == 'pembroke':
        return 'corgi'
    return keyword


def translate_en2ja(keyword):
    url = 'https://script.google.com/macros/s/AKfycbzYpmzfVkissZf6UA-Jo2QyC3n8vSBdPuijdKiwbpLKS86cYVQZ/exec'

    payload = {'text': keyword, 'source': 'en', 'target': 'ja'}
    r = requests.get(url, params=payload)
    return r.text


if __name__ == "__main__":
    """ Sample usage. """
    if len(sys.argv) == 1:
        keyword = 'pembroke'
    else:
        keyword = sys.argv[1]

    print(keyword)
    print(search(keyword))
