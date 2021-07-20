import stanza
import pickle
import nltk
from nltk.tokenize import word_tokenize
import os
from nltk.translate.bleu_score import sentence_bleu
import random

#may20/21
# preprocessing functions
# TODO: use the get_bleu_score_from_list in the code

DIR_OUT_TEMP = "/cs/snapless/oabend/tailin/MT/NEW/outputs/temp_outputs/"


def load_model(lang, model_type):
    if model_type == "pos":
        model = stanza.Pipeline(lang, processors='tokenize,pos', tokenize_no_ssplit=True)
    if model_type == "ner":
        model = stanza.Pipeline(lang, processors='tokenize,ner')
    if model_type == "feat":
        model = stanza.Pipeline(lang, processors='tokenize,mwt,pos,lemma,depparse')


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
    with open(filepath, 'r') as txtfile:
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

def preprocess(txt_path,DIR_OUT_TEMP,name_outfile,lang,model_type,mask_type=None, save_pickle = False):
    """
    :return: parsed (stanza) txt, to load (in order to save running time)
    note: I can parse once with stanza for ner, pos, feats, but for the sake of generality in the code I do it seperately
    """

    dict_model_pickle = {} #save the stanza parsed model to load later (save run time)

    if type(txt_path) == list:
        sentences = txt_path

    else:
        sentences = load_sentences(txt_path)

    model = load_model(lang, model_type)

    if mask_type == "pos":

        for i in range(len(sentences)):
            doc = model(sentences[i])
            dict_model_pickle[i] = doc

        if save_pickle is True:
            save_pickle(DIR_OUT_TEMP + name_outfile + "_"+ mask_type + ".txt",dict_model_pickle)

    if mask_type == "ner":

        for i in range(len(sentences)):
            doc = model(sentences[i])
            dict_model_pickle[i] = doc

        if save_pickle is True:
            save_pickle(DIR_OUT_TEMP + name_outfile + "_"+ mask_type + ".txt",dict_model_pickle)

    if mask_type == "feat":

        for i in range(len(sentences)):
            doc = model(sentences[i])
            dict_model_pickle[i] = doc

        if save_pickle is True:
            save_pickle(DIR_OUT_TEMP + name_outfile + "_"+ mask_type + ".txt",dict_model_pickle)

    return dict_model_pickle
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

        #currently I don't use this list because I use a saved ner list, but this can be changed

        for i in range(len(sentences)):
            doc = model(sentences[i])
            for ent in doc.ents:
                print("ent.type", ent.type)
                #mask_list.append(str(ent.type))

        #mask_list = list(set(mask_list))

    if model_type == "feat":

        for sentence in sentences:
            doc = model(sentence)
            for sent in doc.sentences:
                for word in sent.words:
                    if word.feats is not "None":
                        mask_list.append(word.feats.split("|")[0].split("=")[1]) #CHECK


    mask_list = set(mask_list)

    return mask_list


def mask_text(txt_path, parsed_model, mask, mask_type=None):
    """
    :return: masked sentnces + indices of masked sentences (ones containing the mask)
    """

    #masked_ref, indices_ref = mask_text(reference_path, parsed_model_ref, mask, mask_type)
    #masked_candidate, indices_candidate = mask_text(candidate_path, parsed_model_candidate, mask, mask_type)

    if type(txt_path) == list:
        sentences = txt_path

    else:
        sentences = load_sentences(txt_path)

    print('txt_path',txt_path)
    print('len(sentences)',len(sentences))

    #print("len(sentences)", len(sentences))

    masked_sentences = []
    indices = []

    if mask_type == "pos":

        for i in range(len(sentences)):
            doc = parsed_model[i]
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

    if mask_type == "feat":

        """for i in range(len(sentences)):
            doc = parsed_model[i]
            for sent in doc.sentences:
                sentence_temp = []
                for word in sent.words:
                    if word.feats is not None:
                        if word.feats.split("|")[0].split("=")[0] == mask: #Gender/Number
                            sentence_temp.append(word.feats.split("|")[0].split("=")[1]) #Append the feature itself (value for Gender/Number etc)
                            indices.append(i)
                        else:
                            sentence_temp.append(word.text)
                    else:
                        sentence_temp.append(word.text)
                    masked_sentences.append(" ".join(sentence_temp))
        indices = list(set(indices))"""

        #second trial
        for i in range(len(sentences)):
            doc = parsed_model[i]
            for sent in doc.sentences:
                sentence_temp = []
                for word in sent.words:
                    if (word.feats is not None) and (word.feats.split("|")[0].split("=")[0] == mask):
                        #if word.feats.split("|")[0].split("=")[0] == mask:  # Gender/Number
                            sentence_temp.append(word.feats.split("|")[0].split("=")[1])  # Append the feature itself (value for Gender/Number etc)
                            indices.append(i)
                        #else:
                        #    sentence_temp.append(word.text)
                    else:
                        sentence_temp.append(word.text)
            masked_sentences.append(" ".join(sentence_temp))
        indices = list(set(indices))

    if mask_type == "ner":

        for i in range(len(sentences)):
            sentence = sentences[i]
            doc = parsed_model[i]
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

    print('len(masked_sentences)',len(masked_sentences))
    print('$' * 10)

    return masked_sentences, indices


def indices_intersection(indices1, indices2):
    indices = list(set(indices1) & set(indices2))

    return indices


###########################################################
# scoring functions

def score_sentence_bleu(reference_path, candidate_path, DIR_OUT = None, indices=None, save_score=False):
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
    print("The sentence_bleu score is: " + str(score))

    return score,bleu


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

def make_folder(MYDIR):
    """
    :param MYDIR: path to folder (to create)
    """
    # make directory
    CHECK_FOLDER = os.path.isdir(MYDIR)

    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(MYDIR)
        print("created folder : ", MYDIR)

    else:
        print(MYDIR, "folder already exists.")

    return

def choose_by_bleu(year,DIR_OUT,save_score = False):

    #note: I didn't save the specific system that gave the max bleu, if we want we can add this here

    MYDIR = DIR_OUT + "bleu_scores_" + str(year)

    make_folder(MYDIR)

    bleu_dict = {} #bleu_dict[year][src+trg] = highest bleu score for this lang pair
    bleu_dict[year] = {}
    filenames_max_score = []

    #iterate over the files
    ind_dir = year
    current_dir = "/cs/snapless/oabend/borgr/SSMT/data/submissions/wmt" + str(ind_dir) + "/plain/"
    ref_dir = current_dir + "references/"
    candidate_dir = current_dir + "system-outputs/"

    for filename in os.listdir(ref_dir):

        reference_path = ref_dir + filename
        filename_new = filename.split("-")

        print("reference_path", reference_path)

        if filename_new[0].split("filename")[0].split("newstest")[1].isnumeric() == True:

            src = filename_new[1][:2]
            trg = filename_new[1][2:5]

            bleu_dict[year][src + trg] = {}
            list_bleu_per_pair = []
            filenames_per_year = []

            if trg == "en":

                ref_list =os.listdir(candidate_dir + filename_new[0] + "/" + src + "-" + trg + "/")

                for current_ref_path in ref_list:

                    current_candidate_path = candidate_dir + filename_new[0] + "/" + src + "-" + trg + "/" + current_ref_path

                    current_bleu_score = score_sentence_bleu(current_ref_path,current_candidate_path,DIR_OUT = None, indices=None, save_score=False)
                    list_bleu_per_pair.append(current_bleu_score)
                    filenames_per_year.append(current_ref_path)

                    print("current_ref_path, current_candidate_path, current_bleu_score", current_ref_path,current_candidate_path, current_bleu_score)

            max_bleu = max(list_bleu_per_pair)

            filenames_max_score.append(filenames_per_year[list_bleu_per_pair.index(max(list_bleu_per_pair))]) #filename with highest bleu score per current year

            bleu_dict[year][src+trg] = max_bleu

            if save_score is True:
                save_pickle(MYDIR + str(year) +"_max_bleu_dict.txt",bleu_dict)
                save_pickle(MYDIR + str(year) + "_max_bleu_list_filenames.txt", bleu_dict)


    return bleu_dict, filenames_per_year


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

def bleu_from_list(bleu_list,indices):

    score = 0.

    return
######################################################################################################################

# full process (seperately for pos,ner; later I will join them)

def compute_scores(reference_path, candidate_path, DIR_OUT, parsed_model_ref, parsed_model_candidate, model_type, lang, mask_list_path, mask_type=None):

    #note: parsed_model_ref & parsed_model_candidate are dictionaries, they are also saved so I can load with pickle, but I didn't do this here

    dict_bleu = {}
    dict_hal = {}
    dict_results = {}

    ref_sentences = load_sentences(reference_path) #not using this
    candidate_sentences = load_sentences(candidate_path) #not using this
    print("@@@")
    print("len(ref_sentences),len(candidate_sentences):",len(ref_sentences),len(candidate_sentences))
    print("@@@")

    # compute un-masked bleu score & save to file
    print("no mask -- score_sentence_bleu: ")
    bleu_not_masked, bleu_not_masked_list = score_sentence_bleu(reference_path, candidate_path, DIR_OUT, indices=None, save_score=True)
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

        masked_ref, indices_ref = mask_text(reference_path, parsed_model_ref, mask, mask_type)
        masked_candidate, indices_candidate = mask_text(candidate_path, parsed_model_candidate, mask, mask_type)

        print("@@@")
        print("len(indices_ref),len(indices_candidate): ",len(indices_ref),len(indices_candidate))
        print("len(masked_ref), len(masked_candidate): ", len(masked_ref), len(masked_candidate))
        print("@@@")

        #trial
        save_pickle("/cs/snapless/oabend/tailin/MT/NEW/outputs/"+"masked_ref_"+str(mask)+"_"+str(mask_type)+".txt",masked_ref)
        save_pickle("/cs/snapless/oabend/tailin/MT/NEW/outputs/" + "masked_candidate_" + str(mask) + "_" + str(mask_type) + ".txt",masked_ref)
        ################################################################################################################################

        indices = indices_intersection(indices_ref, indices_candidate)

        print("nu. of indices for "+ str(mask), len(indices))

        #bleu
        #TODO: fix the total_bleu_score func. (for now I flagged False, but it should be True to save running time) [???]
        #total_bleu = total_bleu_score(reference_path, masked_ref, candidate_path,masked_candidate,indices,ref_bleu = False)

        score_masked,_ = score_sentence_bleu(masked_ref, masked_candidate, indices, save_score=False)
        #score_not_masked = score_sentence_bleu(ref_sentences, candidate_sentences, indices, save_score=False) I delete this to save running time (bleu loaded once)
        score_not_masked = 0.
        for i in indices:
            score_not_masked += bleu_not_masked_list[i]
        score_not_masked /= len(indices)

        print("score_masked",score_masked)
        print("score_not_masked",score_not_masked)

        total_bleu = score_masked - score_not_masked

        #total_bleu = score_sentence_bleu(masked_ref, masked_candidate, indices, save_score=False) - score_sentence_bleu(ref_sentences, candidate_sentences, indices, save_score=False)  #not using the list here

        print("total_bleu_score", total_bleu)
        dict_bleu[mask] = total_bleu

        #hallucination
        add, miss, hit = hallucination_score(masked_ref, masked_candidate, mask, indices)
        dict_hal[mask] = [add, miss, hit]

    dict_results["bleu"] = dict_bleu
    dict_results["hallucination"] = dict_hal

    # get filename
    temp = len(os.path.splitext(reference_path)[0].split(".")[0].split("/"))
    temp_filename = os.path.splitext(reference_path)[0].split(".")[0].split("/")[temp - 1]
    print("temp_filename", temp_filename)

    # save_pickle(DIR_OUT + os.path.splitext(reference_path)[0]+"_"+mask_type + "_results_bleu.txt", dict_results)
    save_pickle(DIR_OUT + temp_filename + "_" + mask_type + "_results_bleu.txt", dict_results)

    print("dict_results:")
    print(dict_results)

    return dict_results


######################################################################################################################
######################################################################################################################
######################################################################################################################

# main

DIR_OUT = "/cs/snapless/oabend/tailin/MT/eval/scores/outputs/"

mask_list_path_ner = "/cs/snapless/oabend/tailin/MT/data/en_trials/ner_full_list.txt"
mask_list_path_pos = "/cs/snapless/oabend/tailin/MT/data/en_trials/pos_full_list.txt"
mask_list_path_feat = "/cs/snapless/oabend/tailin/MT/data/en_trials/feats_list.txt"


for ind_dir in range(16, 17):

    current_dir = "/cs/snapless/oabend/borgr/SSMT/data/submissions/wmt" + str(ind_dir) + "/plain/"
    ref_dir = current_dir + "references/"
    candidate_dir = current_dir + "system-outputs/"
    for filename in os.listdir(ref_dir):

        reference_path = ref_dir + filename
        filename_new = filename.split("-")

        print("reference_path", reference_path)

        if filename_new[0].split("filename")[0].split("newstest")[1].isnumeric() == True:

            src = filename_new[1][:2]
            trg = filename_new[1][2:5]

            if trg == "en":

                random_candidate = random.choice(
                    os.listdir(candidate_dir + filename_new[0] + "/" + src + "-" + trg + "/"))

                candidate_path = candidate_dir + filename_new[0] + "/" + src + "-" + trg + "/" + random_candidate

                print("reference_path", reference_path)
                print("candidate_path", candidate_path)

                # pos

                #mask_text(txt_path, parsed_model, mask, mask_type=None)

                dict_parse_ref_pos = preprocess(reference_path,DIR_OUT_TEMP,src+"_dict_parse_ref.txt",lang="en",model_type = "pos",mask_type="pos", save_pickle = False)
                dict_parse_candidate_pos = preprocess(candidate_path, DIR_OUT_TEMP, src + "_dict_parse_candidate.txt",lang="en", model_type="pos", mask_type="pos", save_pickle=False)

                compute_scores(reference_path, candidate_path, DIR_OUT, dict_parse_ref_pos, dict_parse_candidate_pos,model_type="pos", lang="en",mask_list_path=mask_list_path_pos, mask_type="pos")

                #feat
                dict_parse_ref_feat = preprocess(reference_path, DIR_OUT_TEMP, src + "_dict_parse_ref.txt", lang="en",
                                                model_type="feat", mask_type="feat", save_pickle=False)
                dict_parse_candidate_feat = preprocess(candidate_path, DIR_OUT_TEMP, src + "_dict_parse_candidate.txt",
                                                      lang="en", model_type="feat", mask_type="feat", save_pickle=False)

                compute_scores(reference_path, candidate_path, DIR_OUT, dict_parse_ref_feat, dict_parse_candidate_feat,
                               model_type="feat", lang="en", mask_list_path=mask_list_path_feat, mask_type="feat")

                # ner
                dict_parse_ref_ner = preprocess(reference_path,DIR_OUT_TEMP,src+"_dict_parse_ref.txt",lang="en",model_type = "ner",mask_type="ner", save_pickle = False)
                dict_parse_candidate_ner = preprocess(candidate_path, DIR_OUT_TEMP, src + "_dict_parse_candidate.txt",lang="en", model_type="ner", mask_type="ner", save_pickle=False)

                compute_scores(reference_path, candidate_path, DIR_OUT, dict_parse_ref_ner, dict_parse_candidate_ner, model_type="ner", lang="en",mask_list_path=mask_list_path_ner, mask_type="ner")

        else:
            print("didn't find the file you were looking for.")
            pass

    print(reference_path + "Done !")


for ind_dir in range(16, 17):

    current_dir = "/cs/snapless/oabend/borgr/SSMT/data/submissions/wmt" + str(ind_dir) + "/plain/"
    ref_dir = current_dir + "references/"
    candidate_dir = current_dir + "system-outputs/"
    for filename in os.listdir(ref_dir):

        reference_path = ref_dir + filename
        filename_new = filename.split("-")

        print("reference_path", reference_path)

        if filename_new[0].split("filename")[0].split("newstest")[1].isnumeric() == True:

            src = filename_new[1][:2]
            trg = filename_new[1][2:5]

            if trg == "en":

                random_candidate = random.choice(
                    os.listdir(candidate_dir + filename_new[0] + "/" + src + "-" + trg + "/"))

                candidate_path = candidate_dir + filename_new[0] + "/" + src + "-" + trg + "/" + random_candidate

                print("reference_path", reference_path)
                print("candidate_path", candidate_path)


                #feat
                dict_parse_ref_feat = preprocess(reference_path, DIR_OUT_TEMP, src + "_dict_parse_ref.txt", lang="en",
                                                model_type="feat", mask_type="feat", save_pickle=False)
                dict_parse_candidate_feat = preprocess(candidate_path, DIR_OUT_TEMP, src + "_dict_parse_candidate.txt",
                                                      lang="en", model_type="feat", mask_type="feat", save_pickle=False)

                compute_scores(reference_path, candidate_path, DIR_OUT, dict_parse_ref_feat, dict_parse_candidate_feat,
                               model_type="feat", lang="en", mask_list_path=mask_list_path_feat, mask_type="feat")

        else:
            print("didn't find the file you were looking for.")
            pass

    print(reference_path + "Done !")

##################
"""
#sanity check

reference_path = "/cs/snapless/oabend/tailin/MT/NEW/data/ner_trial/ref_ner.txt"
candidate_path = "/cs/snapless/oabend/tailin/MT/NEW/data/ner_trial/candidate_ner.txt"


# pos

#mask_text(txt_path, parsed_model, mask, mask_type=None)

print("pos")

dict_parse_ref_pos = preprocess(reference_path,DIR_OUT_TEMP,"dict_parse_ref.txt",lang="en",model_type = "pos",mask_type="pos", save_pickle = False)
dict_parse_candidate_pos = preprocess(candidate_path, DIR_OUT_TEMP, "dict_parse_candidate.txt",lang="en", model_type="pos", mask_type="pos", save_pickle=False)

compute_scores(reference_path, candidate_path, DIR_OUT, dict_parse_ref_pos, dict_parse_candidate_pos,model_type="pos", lang="en",mask_list_path=mask_list_path_pos, mask_type="pos")

print("*"*100)

# ner
print("ner")
dict_parse_ref_ner = preprocess(reference_path,DIR_OUT_TEMP,"dict_parse_ref.txt",lang="en",model_type = "ner",mask_type="ner", save_pickle = False)
dict_parse_candidate_ner = preprocess(candidate_path, DIR_OUT_TEMP, "dict_parse_candidate.txt",lang="en", model_type="ner", mask_type="ner", save_pickle=False)

compute_scores(reference_path, candidate_path, DIR_OUT, dict_parse_ref_ner, dict_parse_candidate_ner, model_type="ner", lang="en",mask_list_path=mask_list_path_ner, mask_type="ner")

print("*"*100)"""




