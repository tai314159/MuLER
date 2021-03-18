import os
from functools import reduce

from nltk import word_tokenize

LEXICON_DIR = "lexicons"
SENTIMENT_PATH = os.path.join(LEXICON_DIR, "WKWSCISentimentLexicon_v1.1 - WKWSCI sentiment lexicon w POS.csv")

from nltk.corpus import stopwords
import pandas as pd


def read_lexicon(path):
    if path.endswith(".csv"):
        return pd.read_csv(path)

    if path.endswith(".tsv"):
        return pd.read_table(path)


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

    def __init__(self, lang, lexicon_path):
        if len(lang) == 2:
            lang = Scorer.lang_abbr[lang]
        if lang in Scorer.nltk_langs:
            self.stop_words = set(stopwords.words(lang))
        else:
            self.stop_words = {}
        self.lexicon = read_lexicon(lexicon_path)

    def score_text(self, txt: str):
        txt = word_tokenize(txt)
        txt = self.remove_stopwords(txt)
        return sum([self.lexicon[word] for word in txt])

    def remove_stopwords(self, txt):
        return [word for word in txt if word not in self.stop_words]


class SentimentScorer(Scorer):
    def __init__(self, lang, lexicon_path):
        super().__init__(lang, lexicon_path)
        self.lexicon = {row["term"]: row["sentiment"]
                        for _, row in self.lexicon.iterrows()}
        self.negative_effect_window = 3
        self.negative_words = {"barely", "cease", "hardly", "neither", "no", "non", "not", "nothing", "n't", "prevent",
                               "rarely", "seldom", "stop",
                               "unllikely"}  # taken from Lexicon-based sentiment analysis: Comparative evaluation of six sentiment lexicons

    def score_text(self, txt):
        txt = word_tokenize(txt)
        txt = self.remove_stopwords(txt)
        pos = 0
        neg = 0
        inverse = [1]
        for word in txt:
            if word not in self.lexicon:
                continue
            score = self.lexicon[word]
            print(f"word {word} score {score}")
            if word in self.negative_words:
                print(f"negative {word}")
                inverse.append(-1)
                continue
            else:
                inverse.append(1)
            # inverse score if needed
            inverse = inverse[-self.negative_effect_window - 1:]
            score = score * reduce(lambda x, y: x * y, inverse)
            print(f"inverse {inverse} updated score {score}")
            if score > 0:
                pos += 1
            elif score < 0:
                neg += 1
        if neg == pos == 0:
            return 0
        return (pos - neg) / (pos + neg)


if __name__ == '__main__':
    example_sent = """This is a sample sentence, 
                      showing off the stop great words filtration."""

    scorer = SentimentScorer("en", SENTIMENT_PATH)
    score = scorer.score_text(example_sent)
    print(f"score={score}")