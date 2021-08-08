import os
from functools import reduce

from nltk import word_tokenize

from Scorer import Scorer, LEXICON_DIR

SENTIMENT_PATH = os.path.join(LEXICON_DIR, "WKWSCISentimentLexicon_v1.1 - WKWSCI sentiment lexicon w POS.csv")


class SentimentScorer(Scorer):
    def __init__(self, lang, lexicon_path=SENTIMENT_PATH):
        super().__init__(lang, lexicon_path, "term", "sentiment")
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
            # print(f"word {word} score {score}")
            if word in self.negative_words:
                # print(f"negative {word}")
                inverse.append(-1)
                continue
            else:
                inverse.append(1)
            # inverse score if needed
            inverse = inverse[-self.negative_effect_window - 1:]
            score = score * reduce(lambda x, y: x * y, inverse)
            # print(f"inverse {inverse} updated score {score}")
            if score > 0:
                pos += 1
            elif score < 0:
                neg += 1
        if neg == pos == 0:
            return 0
        return (pos - neg) / len(txt)


if __name__ == '__main__':
    example_sent = """This is a sample sentence, 
                      showing off the good exciting stop great words filtration."""

    scorer = SentimentScorer("en", SENTIMENT_PATH)
    score = scorer.score_text(example_sent)
    print(f"score={score}")