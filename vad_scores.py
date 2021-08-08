import os
from functools import reduce

from nltk import word_tokenize

from Scorer import Scorer, LEXICON_DIR, is_float

VAD_PATH = os.path.join(LEXICON_DIR, "NRC-VAD-Lexicon-Aug2018Release", "NRC-VAD-Lexicon-Aug2018Release", "NRC-VAD-Lexicon.txt")


class ValenceScorer(Scorer):
    def __init__(self, lang, lexicon_path=VAD_PATH):
        super().__init__(lang, lexicon_path, filter=lambda row:not is_float(row[1]))


class DominanceScorer(Scorer):
    def __init__(self, lang, lexicon_path):
        super().__init__(lang, lexicon_path, score_col=3, filter=lambda row:not is_float(row[1]))


class ArousalScorer(Scorer):
    def __init__(self, lang, lexicon_path):
        super().__init__(lang, lexicon_path, score_col=2, filter=lambda row:not is_float(row[1]))


if __name__ == '__main__':
    example_sent = """ essentialness This is a sample sentence, 
                      showing off the good exciting stop great words filtration."""

    scorer = ValenceScorer("en", VAD_PATH)
    score = scorer.score_text(example_sent)
    print(f"Valence score={score}")

    scorer = DominanceScorer("en", VAD_PATH)
    score = scorer.score_text(example_sent)
    print(f"Dominance score={score}")

    scorer = ArousalScorer("en", VAD_PATH)
    score = scorer.score_text(example_sent)
    print(f"Arousal score={score}")
