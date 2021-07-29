import os
from functools import reduce

from nltk import word_tokenize

from Scorer import Scorer, LEXICON_DIR

CONCRETENESS_PATH = os.path.join(LEXICON_DIR, "concreteness", "Concreteness_ratings_Brysbaert_et_al_BRM.txt")


class ConcretenessScorer(Scorer):
    def __init__(self, lang, lexicon_path=CONCRETENESS_PATH):
        super().__init__(lang, lexicon_path, score_col=2, filter=lambda row:not row[1].isdigit(), process_row=lambda x:x.lower())

    # def file_to_dict(self, lang, lexicon_path, token_col=0, score_col=1, filter):
    #     self.lexicon = self.read_lexicon(lexicon_path)
    #     lex = {}
    #     for i in range(1, len(self.lexicon)):
    #         row = self.lexicon[i].split()
    #         if not row[1].isdigit():  # ignore multiword expressions
    #             continue
    #         lex[row[0]] = float(row[2])
    #     self.lexicon = lex


if __name__ == '__main__':
    example_sent = """ essentialness This is a sample sentence, 
                      showing off the good exciting stop great words filtration."""

    scorer = ConcretenessScorer("en", CONCRETENESS_PATH)
    score = scorer.score_text(example_sent)
    print(f"score={score}")
