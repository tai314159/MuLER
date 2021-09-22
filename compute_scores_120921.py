import stanza
import pickle
import nltk
from nltk.tokenize import word_tokenize
import os
import os.path
from nltk.translate.bleu_score import sentence_bleu
import random
from datetime import datetime
from rouge_score import rouge_scorer

CACHE = '/cs/snapless/oabend/tailin/MT/NEW/cache/'

#add: deal with files with empty lines (in mask_text)

# stanza.download(model_dir=CACHE)


def load_model(lang, model_type):
    if model_type == "pos":
        model = stanza.Pipeline(lang, processors='tokenize,pos', tokenize_no_ssplit=True,
                                dir=CACHE)
    if model_type == "ner":
        model = stanza.Pipeline(lang, processors='tokenize,ner', dir=CACHE)
    if model_type == "feat":
        model = stanza.Pipeline(lang, processors='tokenize,mwt,pos,lemma,depparse', dir=CACHE)

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
    # with open(filepath, 'r', encoding="utf-8") as txtfile: #switched 'r' to 'rb'

    print("filepath",filepath)

    with open(filepath, 'r') as txtfile:
        for line in txtfile:
            line = line.rstrip("\n")
            temp_list.append(line)
    return temp_list


def load_ref_candidates(ref, candidates):
    # output: ref = list of sentences, candidate = list of lists of sentences

    if type(ref) == list:
        reference = ref

    else:
        reference = load_sentences(ref)

    if type(candidates[0]) == list:
        candidate = candidates  # list of lists of sentences

    else:
        print("candidates",candidates) ###

        candidate = []
        for i in range(len(candidates)):
            print("i",i) ###
            print("candidates[i]", candidates[i]) ###
            candidate.append(load_sentences(candidates[i]))

    return reference, candidate


def count_lines(filepath):

    file = open(filepath, "r")
    Counter = 0

    # Reading from file
    Content = file.read()
    CoList = Content.split("\n")

    for i in CoList:
        if i:
            Counter += 1

    print("This is the number of lines in the file")
    print(Counter)

    return Counter

def save_file(sentences, outpath):
    """
    param sentences: list of sentences
    """
    with open(outpath, "w") as fileout:
        for sentence in sentences:
            fileout.write(sentence + "\n")
    return


def make_folder(MYDIR):
    """
    :param MYDIR: path to folder (to create)
    """
    # make directory
    CHECK_FOLDER = os.path.isdir(MYDIR)

    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        # print("created folder : ", MYDIR)

    # else:
    #     print(MYDIR, "folder already exists.")

    return


def preprocess(txt_path, DIR_OUT, name_outfile, lang, model_type, mask_type=None, save=False):
    """
    txt_path: list of sentences OR path to txt file
    :return: parsed (stanza) txt, to load (in order to save running time)
    note: I can parse once with stanza for ner, pos, feats, but for the sake of generality in the code I do it seperately
    """

    dict_model_pickle = {}  # save the stanza parsed model to load later (save run time)

    if type(txt_path) == list:
        sentences = txt_path

    else:
        sentences = load_sentences(txt_path)

    model = load_model(lang, model_type)

    if mask_type == "pos":

        for i in range(len(sentences)):
            doc = model(sentences[i])
            dict_model_pickle[i] = doc

        if save is True:
            save_pickle(DIR_OUT + name_outfile + "_" + mask_type + ".txt", dict_model_pickle)

    if mask_type == "ner":

        for i in range(len(sentences)):
            doc = model(sentences[i])
            dict_model_pickle[i] = doc

        if save is True:
            save_pickle(DIR_OUT + name_outfile + "_" + mask_type + ".txt", dict_model_pickle)

    if mask_type == "feat":

        for i in range(len(sentences)):
            doc = model(sentences[i])
            dict_model_pickle[i] = doc

        if save is True:
            save_pickle(DIR_OUT + name_outfile + "_" + mask_type + ".txt", dict_model_pickle)

    return dict_model_pickle


###########################################################

def get_mask_list(txt_path, model_type, lang):
    """
    :param sentences: translations/refs (list of sentences)
    :return: list of pos tags (set)
    #NOT UPDATED!!
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

        # currently I don't use this list because I use a saved ner list, but this can be changed

        for i in range(len(sentences)):
            doc = model(sentences[i])
            # for ent in doc.ents:
            #     print("ent.type", ent.type)
            # mask_list.append(str(ent.type))

        # mask_list = list(set(mask_list))

    if model_type == "feat":

        for sentence in sentences:
            doc = model(sentence)
            for sent in doc.sentences:
                for word in sent.words:
                    if word.feats is not "None":
                        mask_list.append(word.feats.split("|")[0].split("=")[1])  # CHECK

    mask_list = set(mask_list)

    return mask_list


def mask_text(txt_path, parsed_model, mask, mask_type=None):
    """
    :return: masked sentnces + indices of masked sentences (ones containing the mask)
    """

    # masked_ref, indices_ref = mask_text(reference_path, parsed_model_ref, mask, mask_type)
    # masked_candidate, indices_candidate = mask_text(candidate_path, parsed_model_candidate, mask, mask_type)

    if type(txt_path) == list:
        sentences = txt_path

    else:
        sentences = load_sentences(txt_path)

    # print('txt_path', txt_path)
    # print('len(sentences)', len(sentences))

    # print("len(sentences)", len(sentences))

    masked_sentences = []
    indices = []

    if mask_type == "pos":

        counter = 0 ###

        for i in range(len(sentences)):
            print("len(sentences)",len(sentences))###
            doc = parsed_model[i]
            for sent in doc.sentences:
                print("*"*100)
                print("len(doc.sentences)",len(doc.sentences))
                print("*" * 100)
                sentence_temp = []
                for word in sent.words:
                    if word.upos == mask:
                        sentence_temp.append(mask)
                        indices.append(i)
                    else:
                        sentence_temp.append(word.text)
                masked_sentences.append(" ".join(sentence_temp))
                counter += 1
        indices = list(set(indices))

    if mask_type == "feat":

        for i in range(len(sentences)):
            doc = parsed_model[i]
            for sent in doc.sentences:
                sentence_temp = []

                for word in sent.words:

                    if (word.feats is not None) and (
                            word.feats.split("|")[0].split("=")[0] == mask.split("_")[0]):

                        if len(mask.split("_")) > 1:

                            # print("yes -- len(mask.split(_)) > 1")  ##
                            # print("mask", mask)
                            # print("mask.split(_)[1]", mask.split("_")[1])
                            # print("word.upos", word.upos)

                            upos_list.append(word.upos)

                            if mask.split("_")[
                                1] == "NOUN":  # mask only intersection of feat with NOUN

                                # print("MASK_NOUN")
                                # print("word.upos", word.upos)

                                # if word.upos == "NOUN":
                                if word.upos == "PRON":
                                    sentence_temp.append(
                                        mask)  # Append the feature itself (value for Gender/Number etc)
                                    indices.append(i)

                                ###CONTINUE!!###

                            if mask.split("_")[
                                1] == "VERB":  # mask only intersection of feat with NOUN

                                # print("MASK_VERB")
                                # print("word.upos", word.upos)

                                if word.upos == "VERB":
                                    sentence_temp.append(
                                        mask)  # Append the feature itself (value for Gender/Number etc)
                                    indices.append(i)

                        ######

                        else:

                            sentence_temp.append(
                                mask)  # Append the feature itself (value for Gender/Number etc)
                            indices.append(i)

                        ######

                    else:

                        sentence_temp.append(word.text)

            masked_sentences.append(" ".join(sentence_temp))

        indices = list(set(indices))

    if mask_type == "ner":

        for i in range(len(sentences)):
            sentence = sentences[i]
            sentence_list = list(sentence)
            # print("sentence", sentence)
            doc = parsed_model[i]

            for ent in doc.ents:

                if ent.type == mask:

                    ent_text = ent.text
                    # print("ent.text", ent.text)

                    ent_list = ent_text.split(" ")
                    # print("ent_list", ent_list)

                    exp = ent.type

                    for ind_ent in range(1, len(ent_list)):
                        exp = exp + " " + ent.type

                    # print("sentence[:ent.start_char]", sentence[:ent.start_char])
                    # print("sentence[ent.end_char:]", sentence[ent.end_char:])
                    # print("exp", exp)

                    sentence = sentence[:ent.start_char] + exp + sentence[ent.end_char:]
                    indices.append(i)  ##here or tab?
            masked_sentences.append(sentence)
            # print("masked sentence", sentence)  ##
        indices = list(set(indices))
        # print("indices", indices)

    # print('len(masked_sentences)', len(masked_sentences))
    # print('$' * 10)

    return masked_sentences, indices


def indices_intersection(indices1, indices2):
    indices = list(set(indices1) & set(indices2))

    return indices


###########################################################
# scoring functions

def score_sentence_bleu(reference_path, candidate_path, DIR_OUT=None, indices=None,
                        save_score=False):
    """
    reference_path = path to masked (or not masked) references file
    candidate_path = path to masked (or not masked) translations file
    indices = list of indices of rows containing the mask (intersection/union)
    return: sentence bleu score (sacredbleu, uniform weights for 1-4-grams)
    """

    if type(reference_path) == str:
        reference = load_sentences(reference_path)

    else:
        reference = reference_path

    if type(candidate_path) == str:

        candidate = load_sentences(candidate_path)

    else:
        candidate = candidate_path

    ###############################################

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
        if save_score == True:  ## I THINK I SHOULD DELETE THIS OPTION since I don't use it in the new code
            bleu = []
            for i in indices:
                score += sentence_bleu([reference[i].strip().split()], candidate[i].strip().split())
                bleu.append(
                    sentence_bleu([reference[i].strip().split()], candidate[i].strip().split()))
                # get filename
                ##temp = len(os.path.splitext(reference_path)[0].split(".")[0].split("/"))
                ##temp_filename = os.path.splitext(reference_path)[0].split(".")[0].split("/")[temp - 1]
                temp_filename = "temp"
                # print("temp_filename", temp_filename)
            # save_pickle(DIR_OUT + os.path.splitext(reference_path)[0] + "sentence_bleu_scores.txt", bleu)
            save_pickle(DIR_OUT + temp_filename + "_sentence_bleu_scores.txt", bleu)
        else:
            bleu = []
            for i in indices:
                score += sentence_bleu([reference[i].strip().split()], candidate[i].strip().split())
                bleu.append(
                    sentence_bleu([reference[i].strip().split()], candidate[i].strip().split()))

                print("reference[i]",reference[i]) ###delete later
                print("reference[i].strip().split()", reference[i].strip().split()) ###delete later

    score /= len(reference)
    # print("The sentence_bleu score is: " + str(score))

    return score, bleu


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


def choose_by_bleu(ref, candidates):
    # load ref & candidates

    if type(ref) == list:
        reference = ref

    else:
        reference = load_sentences(ref)

    if type(candidates[0]) == list:
        candidate = candidates  # list of lists of sentences

    else:
        candidate = []
        for i in range(len(candidates)):
            candidate.append(load_sentences(candidates[i]))

    # compute max bleu

    list_bleu_per_candidate = []

    for candidate1 in candidate:
        current_bleu_score, _ = score_sentence_bleu(reference, candidate1, DIR_OUT=None,
                                                    indices=None, save_score=False)
        list_bleu_per_candidate.append(current_bleu_score)

    max_bleu = max(list_bleu_per_candidate)  # the max bleu score
    index_max = list_bleu_per_candidate.index(
        max(list_bleu_per_candidate))  # index of candidate with max bleu score

    # print("max bleu score is ", max_bleu)

    max_bleu_dict = {}
    max_bleu_dict["max_bleu"] = max_bleu
    max_bleu_dict["index_max"] = index_max

    return max_bleu_dict


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

        # print("add:", add)
        # print("miss:", miss)
        # print("hit", hit)

    return add, miss, hit


def bleu_from_list(bleu_list, indices):
    score = 0.

    return

def score_sentence_rouge(reference_path, candidate_path, DIR_OUT=None, indices=None,save_score=False):
    """
    reference_path = path to masked (or not masked) references file
    candidate_path = path to masked (or not masked) translations file
    indices = list of indices of rows containing the mask (intersection/union)
    return: (sentence) rougeL percision score
    """

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    #scores = scorer.score('The quick brown fox jumps over the lazy dog','The quick brown dog jumps on the log.')

    if type(reference_path) == str:
        reference = load_sentences(reference_path)

    else:
        reference = reference_path

    if type(candidate_path) == str:

        candidate = load_sentences(candidate_path)

    else:
        candidate = candidate_path

    ###############################################

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
        if save_score == True:  ## I THINK I SHOULD DELETE THIS OPTION since I don't use it in the new code
            score_list = []
            for i in indices:
                score += scorer.score(reference[i], candidate[i])['rougeL'][0]
                score_list.append(scorer.score(reference[i], candidate[i])['rougeL'][0])
                temp_filename = "temp"
            save_pickle(DIR_OUT + temp_filename + "_sentence_rouge_scores.txt", score_list)
        else:
            score_list = []
            for i in indices:
                score += scorer.score(reference[i], candidate[i])['rougeL'][0]
                score_list.append(scorer.score(reference[i], candidate[i])['rougeL'][0])

    score /= len(reference)

    return score, score_list

######################################################################################################################

# full process (seperately for pos,ner; later I will join them)

def compute_scores(reference_input, candidate_input, DIR_OUT, parsed_model_ref,
                   parsed_model_candidate, mask_list_path, mask_type=None, score_type = None):
    """
    :param score_type: "bleu"/"rouge" (if None --> score_type == 'bleu')
    """
    # note: parsed_model_ref & parsed_model_candidate are dictionaries, they are also saved they can be loaded with pickle

    #####
    if score_type == 'bleu' or score_type is None:
        score_type = 'bleu'
        score_function = score_sentence_bleu
    else:
        score_function = score_sentence_rouge
    #####

    if type(mask_list_path) == list:
        mask_list = mask_list_path
    else:
        mask_list = load_sentences(mask_list_path)

    dict_bleu = {}
    dict_hal = {}
    dict_results = {}

    bleu_not_masked, bleu_not_masked_list = score_function(reference_input, candidate_input,
                                                                DIR_OUT, indices=None,
                                                                save_score=True)
    dict_results[score_type+"_not_masked"] = bleu_not_masked

    for mask in mask_list:

        masked_ref, indices_ref = mask_text(reference_input, parsed_model_ref, mask, mask_type)
        masked_candidate, indices_candidate = mask_text(candidate_input, parsed_model_candidate,
                                                        mask, mask_type)


        ################################################################################################################################

        indices = indices_intersection(indices_ref, indices_candidate)

        # bleu

        if len(indices) == 0:

            total_bleu = "NA"
        else:
            score_masked, _ = score_function(masked_ref, masked_candidate, indices,
                                                  save_score=False)

            score_not_masked = 0.
            for i in indices:
                score_not_masked += bleu_not_masked_list[i]
            score_not_masked /= len(indices)

            total_bleu = score_masked - score_not_masked

        dict_bleu[mask] = total_bleu

        # hallucination
        add, miss, hit = hallucination_score(masked_ref, masked_candidate, mask, indices)
        dict_hal[mask] = [add, miss, hit]

    dict_results[score_type] = dict_bleu
    dict_results["hallucination"] = dict_hal

    return dict_results


def parse_ref_candidate(DIR_OUT, trg_lang, mask_type, reference, current_candidate, temp_filename):
    # parse reference & candidate if needed

    CHECK_FOLDER_REF = os.path.isfile(
        DIR_OUT + temp_filename + "_" + "_" + trg_lang + "_dict_parse_ref_" + mask_type + ".txt")
    CHECK_FOLDER_CAN = os.path.isfile(
        DIR_OUT + temp_filename + "_" + "_" + trg_lang + "_dict_parse_candidate_" + mask_type + ".txt")

    if not (CHECK_FOLDER_REF & CHECK_FOLDER_CAN):

        # print("parsing" + "_" + trg_lang + " for " + mask_type + " !")

        dict_parse_ref_mask = preprocess(reference, DIR_OUT,
                                         temp_filename + "_" + "_" + trg_lang + "_dict_parse_ref",
                                         lang=trg_lang,
                                         model_type=mask_type, mask_type=mask_type, save=True)

        dict_parse_candidate_mask = preprocess(current_candidate, DIR_OUT,
                                               temp_filename + "_" + "_" + trg_lang + "_dict_parse_candidate",
                                               lang=trg_lang, model_type=mask_type,
                                               mask_type=mask_type,
                                               save=True)
    else:

        dict_parse_ref_mask = load_pickle(
            DIR_OUT + temp_filename + "_" + trg_lang + "_dict_parse_ref_" + mask_type + ".txt")

        dict_parse_candidate_mask = load_pickle(
            DIR_OUT + temp_filename + "_" + trg_lang + "_dict_parse_candidate_" + mask_type + ".txt")

    return dict_parse_ref_mask, dict_parse_candidate_mask


def generate_filename():
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d:%m:%Y:%H:%M:%S")
    return dt_string


######################################################################################################################
# main function

def run_main(DIR_OUT, mask_list_path, ref_input, candidates_input, mask_type, trg_lang="en",
             max_bleu_dict_path=None, run_all=False,score_type = None):
    """
    :param DIR_OUT:
    :param mask_list_path: path to txt file (each row is a mask) -- OR mask list
    :param ref_input: list of sentences OR path to reference file
    :param candidates_input: list of lists of sentences OR list of lists of candidate paths
    :param mask_type: "pos"/"ner"/"feat"
    :param trg_lang: target lang (which is now only ENGLISH with our current models)
    :param max_bleu_dict_path: path to max_bleu_dict in the same form choose_by_bleu function produce, if exists

    :return: final results in dictionary form and output to a file named "DIR_OUT + mask_type + "_results_bleu.txt""
    """

    # load reference & candidate files
    # NOTE: if the ref OR candidates are of the wrong form, no error message will appear

    reference, candidates = load_ref_candidates(ref_input, candidates_input)

    #####################$$$$$$$$$$$$$$$$$$%%%%%%%%%%%%%%%%^^^^^^^^^^^^^^^^^^

    if run_all is True:

        # run on all candidate files, output a list of results dictionary (i-th element is the results dict for the i-th candidate)

        list_results_dictionary = []

        for ind_candidate in range(len(candidates)):
            current_candidate = candidates[ind_candidate]
            print('current_candidate', len(current_candidate), len(candidates))
            print('reference', len(reference))
            temp_filename = generate_filename()  # generate temporary filename, to save parsed dictionaries

            dict_parse_ref_mask, dict_parse_candidate_mask = parse_ref_candidate(DIR_OUT, trg_lang,
                                                                                 mask_type,
                                                                                 reference,
                                                                                 current_candidate,
                                                                                 temp_filename)  # HERE

            # print("dict_parse_ref_mask.keys()", dict_parse_ref_mask.keys())
            # print("dict_parse_candidate_mask.keys()", dict_parse_candidate_mask.keys())

            ###CONTINUE###

            dict_results = compute_scores(reference, current_candidate, DIR_OUT,
                                          dict_parse_ref_mask,
                                          dict_parse_candidate_mask,
                                          mask_list_path, mask_type)

            # list_results_dictionary[ind_candidate] = dict_results # changed
            list_results_dictionary.append(dict_results)  # changed

        return list_results_dictionary

    else:

        # compute max bleu score & save (or load if exists)

        if max_bleu_dict_path is None:

            max_bleu_dict = choose_by_bleu(ref_input, candidates)

            index_max = max_bleu_dict["index_max"]

        else:

            max_bleu_dict = load_pickle(max_bleu_dict_path)

            index_max = max_bleu_dict["index_max"]

        current_candidate = candidates[index_max]  # this is the candidate with max bleu

        temp_filename = generate_filename()  # generate temporary filename, to save parsed dictionaries

        dict_parse_ref_mask, dict_parse_candidate_mask = parse_ref_candidate(DIR_OUT, trg_lang,
                                                                             mask_type,
                                                                             reference,
                                                                             current_candidate,
                                                                             temp_filename)
        # now we can use the parsed files instead of parsing at each iteration
        # compute scores (for each mask_type)

        print("*" * 200)
        print("len(reference)", len(reference))
        print("*"*200)
        print("len(current_candidate)",len(current_candidate))
        print("*" * 200)

        dict_results = compute_scores(reference, current_candidate, DIR_OUT, dict_parse_ref_mask,
                                      dict_parse_candidate_mask,
                                      mask_list_path, mask_type,score_type)

        return dict_results

######################################################################################################################
######################################################################################################################
######################################################################################################################

# main

# ARGS
# DIR_OUT = "..."
#
# mask_type = "ner"
#
# mask_list_path = "/cs/snapless/oabend/tailin/MT/NEW/codes/trial_data/"+mask_type+"_mask_list.txt"
#
# ref = "/cs/snapless/oabend/tailin/MT/NEW/codes/trial_data/ref_ner_neg.txt"
#
# candidates = ["/cs/snapless/oabend/tailin/MT/NEW/codes/trial_data/can_ner_neg.txt"]

# ARGS
DIR_OUT = "/cs/snapless/oabend/tailin/MT/NEW/outputs/11.08.21/" #CHANGE

#mask_type = "pos"

#mask_list_path = "/cs/snapless/oabend/tailin/MT/NEW/codes/trial_data/"+mask_type+"_mask_list.txt"

#ref_input = "/cs/snapless/oabend/tailin/MT/NEW/codes/trial_data/ref.txt"
#candidates_input = ["/cs/snapless/oabend/tailin/MT/NEW/codes/trial_data/candidate.txt"]

#ref_input = "/cs/snapless/oabend/borgr/SSMT/data/submissions/wmt20/plain/references/newstest2020-deen-ref.en.txt"
#candidates_input = ["/cs/snapless/oabend/borgr/SSMT/data/submissions/wmt20/plain/system-outputs/de-en/newstest2020.de-en.yolo.1052.txt"]

###############################################

#summarization
ref_input = "/cs/snapless/oabend/tailin/MT/MT_eval/summarization_data/ref/test.target.ref.txt"
candidates_input = ["/cs/snapless/oabend/tailin/MT/MT_eval/summarization_data/candidates/t5_base_zs.txt"]



mask_types = ["pos","feat","ner"]
#mask_types = ["pos"]

for mask_type in mask_types:
    mask_list_path = "/cs/snapless/oabend/tailin/MT/NEW/codes/trial_data/" + mask_type + "_mask_list.txt"
    result = run_main(DIR_OUT, mask_list_path, ref_input, candidates_input,mask_type, trg_lang = "en",max_bleu_dict_path = None, run_all = False, score_type = "rouge")
    print("result")
    print(result)