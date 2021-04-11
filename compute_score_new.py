import stanza
import pickle
import nltk
from nltk.tokenize import word_tokenize
import os
from nltk.translate.bleu_score import sentence_bleu

# preprocessing functions
#TODO: use the get_bleu_score_from_list in the code

def load_model(lang, model_type):
    if model_type == "pos":
        model = stanza.Pipeline(lang, processors='tokenize,pos', tokenize_no_ssplit=True)
    if model_type == "ner":
        model = stanza.Pipeline(lang, processors='tokenize,ner')

    return model


def load_pickle(filepath):
    with open(filepath, 'rb') as file:
        loaded_file = pickle.load(file)
    return loaded_file


def save_pickle(out_filepath, file_to_save):
    with open(out_filepath, 'wb') as fp:
        pickle.dump(file_to_save, fp)


def load_sentences(filepath):
    """
    :param filepath: filepath to translatons/refs
    :return: list of sentences
    """
    temp_list = []
    with open(filepath, 'r', encoding="utf-8") as txtfile:
        for line in txtfile:
            line = line.rstrip("\n")
            temp_list.append(line)
    return temp_list


def save_file(sentences, outpath):
    """
    param sentences: list of sentences
    """
    with open(outpath, "w") as fileout:
        for sentence in sentences:
            fileout.write(sentence + "\n")
    return


###########################################################

# def get_mask_list(txt_path, model, pos = False, ner = False):
def get_mask_list(txt_path, model_type, lang):
    """
    :param sentences: translations/refs (list of sentences)
    :return: list of pos tags (set)
    """
    model = load_model(lang, model_type)
    sentences = load_sentences(txt_path)

    mask_list = []

    if model_type == "pos":

        for sentence in sentences:
            doc = model(sentence)
            for sent in doc.sentences:
                for word in sent.words:
                    mask_list.append(word.upos)

    mask_list = set(mask_list)

    if model_type == "ner":

        for i in range(len(sentences)):
            doc = model(sentences[i])
            for ent in doc.ents:
                print("ent.type",ent.type)
                #mask_list.append(str(ent.type))

        #mask_list = list(set(mask_list))

    return mask_list

# def mask_text(txt_path, model, pos = None, ner = None, mask = None):

def mask_text(txt_path, model, mask, mask_type=None):
    if type(txt_path) == list:
        sentences = txt_path

    else:
        sentences = load_sentences(txt_path)

    # print("len(sentences)", len(sentences))

    masked_sentences = []
    indices = []

    if mask_type == "pos":

        for i in range(len(sentences)):
            doc = model(sentences[i])
            for sent in doc.sentences:
                sentence_temp = []
                for word in sent.words:
                    if word.upos == mask:
                        sentence_temp.append(mask)
                        indices.append(i)
                    else:
                        sentence_temp.append(word.text)
                masked_sentences.append(" ".join(sentence_temp))
        indices = list(set(indices))

    if mask_type == "ner":

        for i in range(len(sentences)):
            sentence = sentences[i]
            doc = model(sentence)
            for ent in doc.ents:
                if ent.type == mask:
                    sentence = sentence.replace(ent.text, ent.type)
                    indices.append(i)
            masked_sentences.append(sentence)
        indices = list(set(indices))

    """else:

        for i in range(len(sentences)): 
            sentence = word_tokenize(sentences[i])

            #masked_sentences.append(" ".join(sentence[i].replace(mask,"MASK")))
            indices.append(i)
        indices = list(set(indices))"""

    return masked_sentences, indices

def indices_intersection(indices1, indices2):
    indices = list(set(indices1) & set(indices2))

    return indices
###########################################################
# scoring functions

def score_sentence_bleu(reference_path, candidate_path, indices=None, save_score=False):
    """
    reference_path = path to masked (or not masked) references file
    trans_path = path to masked (or not masked) translations file
    indices = list of indices of rows containing the mask (intersection/union)
    return: sentence belu score (sacredbleu, uniform weights for 1-4-grams)
    """
    # TODO: add here out directory

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
            save_pickle(os.path.splitext(reference_path)[0] + "sentence_bleu_scores.txt", bleu)
        else:
            for i in indices:
                score += sentence_bleu([reference[i].strip().split()], candidate[i].strip().split())

    score /= len(reference)
    print("The sentence_bleu score is: " + str(score))

    return score


def get_bleu_score_from_list(list_bleu, indices=None):
    """
    compute bleu score, given indices, from saved bleu scores list
    """
    if indices == None:  # this will compute the regular sentence-bleu
        indices = range(len(reference))
    score = 0.
    for i in indices:
        score += list_bleu[i]

    return score


def hallucination_score(reference_path, candidate_path, mask, indices=None):
    """
    mask = str
    """
    if type(reference_path) == list:
        reference = reference_path
        candidate = candidate_path

    else:
        reference = load_sentences(reference_path)
        candidate = load_sentences(candidate_path)

    if indices == None:
        indices = range(len(reference))

    if len(indices) == 0:
        add = 0
        miss = 0
        hit = 0
    else:
        add = 0
        miss = 0
        hit = 0
        for i in indices:

            ref_row = word_tokenize(reference[i])
            candidate_row = word_tokenize(candidate[i])

            if ref_row.count(mask) > candidate_row.count(mask):
                add += 1
            if ref_row.count(mask) < candidate_row.count(mask):
                miss += 1
            if ref_row.count(mask) == candidate_row.count(mask):
                hit += 1

        add /= len(indices)
        miss /= len(indices)
        hit /= len(indices)

        print("add:", add)
        print("miss:", miss)
        print("hit", hit)

    return add, miss, hit

###########################################################

# full process (seperately for pos,ner; later I will join them)

def compute_scores(reference_path,candidate_path,DIR_OUT,model_type,lang,mask_list_path,mask_type=None):
    dict_bleu = {}
    dict_hal = {}
    dict_results= {}

    ref_sentences = load_sentences(reference_path)
    candidate_sentences = load_sentences(candidate_path)

    # compute un-masked bleu score & save to file
    print("no mask -- score_sentence_bleu: ")
    bleu_not_masked = score_sentence_bleu(reference_path, candidate_path, indices=None, save_score=True)
    dict_results["bleu_not_masked"] = bleu_not_masked
    #####################################
    # pos

    model = load_model(lang, model_type)
    mask_list = load_sentences(mask_list_path)  # TODO: or generate it with get_pos_list func

    print("mask_list", mask_list)  ###

    # compute un-masked bleu score & save to file
    # score_sentence_bleu(reference_path, candidate_path ,indices=None, save_score = True)

    for mask in mask_list:

        print("mask_type", mask)

        masked_ref, indices_ref = mask_text(reference_path, model, mask, mask_type)
        masked_candidate, indices_candidate = mask_text(candidate_path, model, mask, mask_type)
        indices = indices_intersection(indices_ref, indices_candidate)

        # bleu
        # TODO: fix the total_bleu_score func. (for now I flagged False, but it should be True to save running time)
        # total_bleu = total_bleu_score(reference_path, masked_ref, candidate_path,masked_candidate,indices,ref_bleu = False)

        total_bleu = score_sentence_bleu(masked_ref, masked_candidate, indices, save_score=False) - score_sentence_bleu(ref_sentences, candidate_sentences,indices,save_score=False)  # not using the list here

        print("total_bleu_score", total_bleu)
        dict_bleu[mask] = total_bleu

        # hallucination
        add, miss, hit = hallucination_score(masked_ref, masked_candidate, mask, indices)
        dict_hal[mask] = [add, miss, hit]

    dict_results["bleu"] = dict_bleu
    dict_results["hallucination"] = dict_hal

    save_pickle(os.path.splitext(reference_path)[0]+"_"+mask_type + "_results_bleu.txt", dict_results)

    print("dict_results:")
    print(dict_results)

    return dict_results

###############################################################

# args
DIR_DATA = "/cs/snapless/oabend/tailin/MT/data/en_trials/"
DIR_OUT = "/cs/snapless/oabend/tailin/MT/data/en_trials/"
DIR_DUMP = "/cs/snapless/oabend/tailin/MT/data/en_trials/"

mask_list_path_ner  = DIR_DATA + "ner_full_list.txt"
mask_list_path_pos = DIR_DATA + "pos_full_list.txt"

reference_path = DIR_DATA+"refs_long.txt"
candidate_path = DIR_DATA+"refs_long.txt"

###############################################################

#pos
compute_scores(reference_path,candidate_path,DIR_OUT,model_type="pos",lang = "en",mask_list_path = mask_list_path_pos,mask_type="pos")

#ner
compute_scores(reference_path,candidate_path,DIR_OUT,model_type="ner",lang="en",mask_list_path = mask_list_path_ner,mask_type="ner")


