import stanza
import pickle
import nltk
from nltk.tokenize import word_tokenize
import os
import os.path
from nltk.translate.bleu_score import sentence_bleu
import random


def score_sentence_bleu(reference_path, candidate_path, DIR_OUT=None, indices=None, save_score=False):
    """
    reference_path = path to masked (or not masked) references file
    candidate_path = path to masked (or not masked) translations file
    indices = list of indices of rows containing the mask (intersection/union)
    return: sentence bleu score (sacredbleu, uniform weights for 1-4-grams)
    """

    # if type(reference_path) == str & type(candidate_path) == str :
    if type(reference_path) == str:
        reference = load_sentences(reference_path)
        candidate = load_sentences(candidate_path)

    else:
        reference = reference_path
        candidate = candidate_path

    if len(reference) == 0:
        raise ValueError('The reference/candidate is of length zero.')

    score = 0.

    if indices == None:  # this will compute the regular sentence-bleu
        indices = range(len(reference))

    if len(reference) != len(candidate):
        raise ValueError('The number of sentences in both files do not match.')

    if len(indices) == 0:
        score = 0.
    else:
        if save_score == True:
            bleu = []
            for i in indices:
                score += sentence_bleu([reference[i].strip().split()], candidate[i].strip().split())
                bleu.append(sentence_bleu([reference[i].strip().split()], candidate[i].strip().split()))
                # get filename
                temp = len(os.path.splitext(reference_path)[0].split(".")[0].split("/"))
                temp_filename = os.path.splitext(reference_path)[0].split(".")[0].split("/")[temp - 1]
                print("temp_filename", temp_filename)
            # save_pickle(DIR_OUT + os.path.splitext(reference_path)[0] + "sentence_bleu_scores.txt", bleu)
            save_pickle(DIR_OUT + temp_filename + "_sentence_bleu_scores.txt", bleu)
        else:
            bleu = []
            for i in indices:
                score += sentence_bleu([reference[i].strip().split()], candidate[i].strip().split())
                bleu.append(sentence_bleu([reference[i].strip().split()], candidate[i].strip().split()))
    score /= len(reference)
    # print("The sentence_bleu score is: " + str(score))

    return score, bleu