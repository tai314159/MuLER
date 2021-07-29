import os
from functools import reduce

from nltk import word_tokenize
import os

LEXICON_DIR = os.path.join(os.path.dirname(__file__), "lexicons")
SENTIMENT_PATH = os.path.join(LEXICON_DIR, "WKWSCISentimentLexicon_v1.1 - WKWSCI sentiment lexicon w POS.csv")

from nltk.corpus import stopwords
import pandas as pd
import numpy as np


class Scorer:
    lang_abbr = {'hu': 'hungarian',
                 'sw': 'swedish',
                 'ka': 'kazakh',
                 'no': 'norwegian',
                 'fi': 'finnish',
                 'ar': 'arabic',
                 'in': 'indonesian',
                 'po': 'portuguese',
                 'tu': 'turkish',
                 'az': 'azerbaijani',
                 'sl': 'slovene',
                 'sp': 'spanish',
                 'da': 'danish',
                 'ne': 'nepali',
                 'ro': 'romanian',
                 'gr': 'greek',
                 'du': 'dutch',
                 'ta': 'tajik',
                 'ge': 'german',
                 'en': 'english',
                 'ru': 'russian',
                 'fr': 'french',
                 'it': 'italian'}
    nltk_langs = [val for val in lang_abbr.values()]

    def __init__(self, lang, lexicon_path, token_col=0, score_col=1, filter=lambda x: False, process_row=lambda x: x,
                 filter_stopwords=True):
        if len(lang) == 2:
            lang = Scorer.lang_abbr[lang]
        if filter_stopwords and lang in Scorer.nltk_langs:
            self.stop_words = set(stopwords.words(lang))
        else:
            self.stop_words = {}
        self.lexicon = self.file_to_dict(lang, lexicon_path, token_col, score_col, filter, process_row)

    def file_to_dict(self, lang, lexicon_path, token_col=0, score_col=1, filter=lambda x: False,
                     process_row=lambda x: x):
        lexicon = self.read_lexicon(lexicon_path)
        if isinstance(lexicon, pd.DataFrame):  # assume a df
            lexicon.apply(process_row, axis=1)
            lexicon = {row[token_col]: row[score_col]
                       for _, row in lexicon.iterrows() if not filter(row)}
            return lexicon
        else:  # spaces\tabs separated lines
            dct = {}
            for i in range(1, len(lexicon)):
                row = process_row(lexicon[i]).split()
                if filter(row):
                    continue
                dct[row[token_col]] = float(row[score_col])
            return dct

    def score_text(self, txt: str):
        txt = word_tokenize(txt)
        txt = self.remove_stopwords(txt)
        scores = [self.lexicon[word] for word in txt if word in self.lexicon]
        if not scores:
            return 0
        # print(f"scores {scores}")
        return np.mean(scores)

    def score_batch(self, batch: list, reduce=False):
        """
        Get a score for a batch of instances
        :param batch: list of str sentences
        :param reduce: wether to return a mean (when True) of a list of scores (when False)
        """
        scores = []
        for txt in batch:
            if type(txt) != str:
                raise RuntimeError('Batch elements should be of type str')
            scores.append(self.score_text(txt))
        if reduce:
            return np.mean(scores)
        return np.array(scores)

    def remove_stopwords(self, txt):
        return [word for word in txt if word not in self.stop_words]

    @staticmethod
    def read_lexicon(path):
        if not path:
            return
        if path.endswith(".csv"):
            return pd.read_csv(path)
        if path.endswith(".tsv"):
            return pd.read_table(path)
        if path.endswith(".xlsx"):
            df = pd.read_excel(path)
            return df
        if path.endswith(".txt"):
            with open(path) as fl:
                res = fl.readlines()
            return res

        print("File extension not recognized, assuming text file")
        with open(path) as fl:
            res = fl.readlines()
        return res

def is_float(value):
  try:
    float(value)
    return True
  except ValueError:
    return False
