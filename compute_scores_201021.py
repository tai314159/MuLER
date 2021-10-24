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
import re

CACHE = '/cs/snapless/oabend/tailin/MT/NEW/cache/'

# stanza.download(model_dir=CACHE)


"""def load_model(lang, model_type):
    if model_type == "pos":
        model = stanza.Pipeline(lang, processors='tokenize,pos', tokenize_no_ssplit=True,
                                dir=CACHE)
    if model_type == "ner":
        model = stanza.Pipeline(lang, processors='tokenize,ner', dir=CACHE)
    if model_type == "feat":
        model = stanza.Pipeline(lang, processors='tokenize,mwt,pos,lemma,depparse', dir=CACHE)

    return model"""

"""def load_model(lang, model_type):
    if model_type == "pos":
        model = stanza.Pipeline(lang, processors='tokenize,pos', tokenize_no_ssplit=True)
    if model_type == "ner":
        model = stanza.Pipeline(lang, processors='tokenize,ner')
    if model_type == "feat":
        #model = stanza.Pipeline(lang, processors='tokenize,mwt,pos,lemma,depparse')
        model = stanza.Pipeline(lang, processors='tokenize,pos,depparse')

    return model"""


def load_model(lang, model_type):
    if model_type == "pos":
        model = stanza.Pipeline(lang, processors='tokenize,pos,lemma', tokenize_no_ssplit=True)
    if model_type == "ner":
        model = stanza.Pipeline(lang, processors='tokenize,ner')
    if model_type == "feat":
        # model = stanza.Pipeline(lang, processors='tokenize,mwt,pos,lemma,depparse')
        model = stanza.Pipeline(lang, processors='tokenize,pos,depparse,lemma')

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

    ##print("filepath",filepath)

    with open(filepath, 'r') as txtfile:
        for line in txtfile:
            line = line.rstrip("\n")
            temp_list.append(line)
    return temp_list


def countX(lst, x):
    count = 0
    for ele in lst:
        if (ele == x):
            count = count + 1
    return count


def load_ref_candidates(ref, candidates):
    # output: ref = list of sentences, candidate = list of lists of sentences

    if type(ref) == list:
        reference = ref

    else:
        reference = load_sentences(ref)

    if type(candidates[0]) == list:
        candidate = candidates  # list of lists of sentences

    else:
        ##print("candidates",candidates) ###

        candidate = []
        for i in range(len(candidates)):
            ##print("i",i) ###
            ##print("candidates[i]", candidates[i]) ###
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

    ##print("This is the number of lines in the file")
    ##print(Counter)

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


def mask_text_old(txt_path, parsed_model, mask, mask_type=None):
    """
    :return: masked sentnces + indices of masked sentences (ones containing the mask)
    """

    if type(txt_path) == list:
        sentences = txt_path

    else:
        sentences = load_sentences(txt_path)

    masked_sentences = []
    indices = []

    if mask_type == "pos":

        for i in range(len(sentences)):

            doc = parsed_model[i]

            if not sentences[i]:
                # print("empty str")
                sentence_temp = []
                masked_sentences.append(" ".join(sentence_temp))
            else:
                # print("not empty str")
                for sent in doc.sentences:
                    ##print("sent",sent) ##

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

        for i in range(len(sentences)):
            doc = parsed_model[i]
            for sent in doc.sentences:
                sentence_temp = []
                ##print("sent",sent)###

                ##print("sent.words",sent.words)
                for word in sent.words:
                    ##print("word",word)
                    if (word.feats is not None) and (
                            word.feats.split("|")[0].split("=")[0] == mask.split("_")[0]):

                        if len(mask.split("_")) > 1:  # if we split the mask by pos types

                            upos_list.append(word.upos)

                            if mask.split("_")[
                                1] == "NOUN":  # mask only intersection of feat with NOUN

                                if word.upos == "PRON":
                                    sentence_temp.append(
                                        mask)  # Append the feature itself (value for Gender/Number etc)
                                    indices.append(i)

                            if mask.split("_")[
                                1] == "VERB":  # mask only intersection of feat with NOUN

                                if word.upos == "VERB":
                                    sentence_temp.append(
                                        mask)  # Append the feature itself (value for Gender/Number etc)
                                    indices.append(i)

                        ######

                        else:  # regular case (feat not splitted to pos types)

                            sentence_temp.append(
                                mask)  # Append the feature itself (value for Gender/Number etc)
                            indices.append(i)
                            ##print("mask",mask)

                        ######

                    else:

                        sentence_temp.append(word.text)
                        ##print("word.text",word.text)

            masked_sentences.append(" ".join(sentence_temp))
            ##print(".join(sentence_temp)"," ".join(sentence_temp))###

        indices = list(set(indices))

    if mask_type == "ner":

        for i in range(len(sentences)):
            sentence = sentences[i]
            doc = parsed_model[i]

            ner_doc_final = []  ##

            for ent in doc.ents:

                # print("doc.ents",doc.ents)
                # print("ent.start_char",ent.start_char)
                # print("ent.end_char", ent.end_char)

                if ent.type == mask:

                    ner_doc = []

                    ner_doc.append(ent.start_char)
                    ner_doc.append(ent.end_char)

                    ent_text = ent.text
                    ##print("ent.text", ent.text)

                    ent_list = ent_text.split(" ")
                    ##print("ent_list", ent_list)

                    exp = ent.type

                    for ind_ent in range(1, len(ent_list)):
                        exp = exp + " " + ent.type

                    ner_doc.append(exp)

                    ##print("sentence[:ent.start_char]", sentence[:ent.start_char])
                    ##print("sentence[ent.end_char:]", sentence[ent.end_char:])
                    ##print("exp", exp)

                    ##sentence = sentence[:ent.start_char] + exp + sentence[ent.end_char:]
                    indices.append(i)  ##here or tab?

                    ner_doc_final.append(ner_doc)

            #####

            new_sentence = ''
            if len(ner_doc_final) == 0:
                new_sentence = sentence
            else:
                new_sentence = sentence[0:ner_doc_final[0][0]]
                num_of_ent = len(ner_doc_final)
                for i in range(num_of_ent):
                    new_sentence += ner_doc_final[i][2]
                    if i != len(ner_doc_final) - 1:
                        new_sentence += sentence[ner_doc_final[i][1]: ner_doc_final[i + 1][0]]
                    else:
                        new_sentence += sentence[ner_doc_final[i][1]:]

            # for key in ner_dict_doc.keys():

            masked_sentences.append(new_sentence)

            ##print("masked sentence", new_sentence)  ##
        indices = list(set(indices))
        # print("indices", indices)

    # print('len(masked_sentences)', len(masked_sentences))
    # print('$' * 10)

    return masked_sentences, indices


def mask_text_version0(txt_path, parsed_model, mask, mask_type=None):
    """
    :return: masked sentnces + indices of masked sentences (ones containing the mask)
    """

    print("version 0") ###

    if type(txt_path) == list:
        sentences = txt_path

    else:
        sentences = load_sentences(txt_path)

    masked_sentences = []
    indices = []

    if mask_type == "pos":

        for i in range(len(sentences)):

            doc = parsed_model[i]

            if not sentences[i]:
                sentence_temp = []
                masked_sentences.append(" ".join(sentence_temp))
            else:
                for sent in doc.sentences:

                    sentence_temp = []
                    for word in sent.words:
                        if word.upos == mask:
                            sentence_temp.append(mask)
                            indices.append(i)
                        else:
                            sentence_temp.append(word.text)

                    if mask + "'" in sentence_temp:
                        sentence_temp = [mask if x == mask + "'" else x for x in sentence_temp]
                    if mask + "'s" in sentence_temp:
                        sentence_temp = [mask if x == mask + "'s" else x for x in sentence_temp]

                    new_sentence = " ".join(sentence_temp)


                    masked_sentences.append(new_sentence)
        indices = list(set(indices))

    if mask_type == "feat":

        for i in range(len(sentences)):
            doc = parsed_model[i]
            for sent in doc.sentences:
                sentence_temp = []

                for word in sent.words:
                    ##print("word",word)
                    if (word.feats is not None) and (
                            word.feats.split("|")[0].split("=")[0] == mask.split("_")[0]):

                        if len(mask.split("_")) > 1:  # if we split the mask by pos types

                            upos_list.append(word.upos)

                            if mask.split("_")[
                                1] == "NOUN":  # mask only intersection of feat with NOUN

                                if word.upos == "PRON":
                                    sentence_temp.append(
                                        mask)  # Append the feature itself (value for Gender/Number etc)
                                    indices.append(i)

                            if mask.split("_")[
                                1] == "VERB":  # mask only intersection of feat with NOUN

                                if word.upos == "VERB":
                                    sentence_temp.append(
                                        mask)  # Append the feature itself (value for Gender/Number etc)
                                    indices.append(i)

                        ######

                        else:  # regular case (feat not splitted to pos types)

                            sentence_temp.append(
                                mask)  # Append the feature itself (value for Gender/Number etc)
                            indices.append(i)

                        ######

                    else:

                        sentence_temp.append(word.text)

            if mask + "'" in sentence_temp:
                sentence_temp = [mask if x == mask + "'" else x for x in sentence_temp]
            if mask + "'s" in sentence_temp:
                sentence_temp = [mask if x == mask + "'s" else x for x in sentence_temp]

            new_sentence = " ".join(sentence_temp)
            masked_sentences.append(new_sentence)

        indices = list(set(indices))

    if mask_type == "ner":

        for i in range(len(sentences)):
            sentence = sentences[i]
            doc = parsed_model[i]

            ner_doc_final = []  ##

            for ent in doc.ents:


                if ent.type == mask:

                    if ent.text.startswith("the"):

                        print("ent.text.startswith(the)")

                        ner_doc = []

                        ner_doc.append(ent.start_char + 4)
                        ner_doc.append(ent.end_char)

                        ent_text = ent.text
                        print("ent.text", ent_text)
                        ent_text = ent_text.strip("the ")
                        print("ent_text.strip(the)", ent.text)

                        ent_list = ent_text.split(" ")

                        exp = ent.type

                        for ind_ent in range(1, len(ent_list)):
                            exp = exp + " " + ent.type

                        ner_doc.append(exp)

                        indices.append(i)  ##here or tab?

                        ner_doc_final.append(ner_doc)



                    else:
                        ner_doc = []

                        ner_doc.append(ent.start_char)
                        ner_doc.append(ent.end_char)

                        ent_text = ent.text
                        print("ent.text", ent.text)
                        print("ent.start_char", ent.start_char)
                        print("ent.end_char", ent.end_char)

                        ent_list = ent_text.split(" ")

                        exp = ent.type

                        for ind_ent in range(1, len(ent_list)):
                            exp = exp + " " + ent.type

                        ner_doc.append(exp)

                        indices.append(i)  ##here or tab?

                        ner_doc_final.append(ner_doc)
                        #####

            #####

            print("ner_doc_final", ner_doc_final)

            new_sentence = ''
            if len(ner_doc_final) == 0:
                new_sentence = sentence
            else:
                new_sentence = sentence[0:ner_doc_final[0][0]]
                num_of_ent = len(ner_doc_final)
                for i in range(num_of_ent):
                    new_sentence += ner_doc_final[i][2]
                    if i != len(ner_doc_final) - 1:
                        new_sentence += sentence[ner_doc_final[i][1]: ner_doc_final[i + 1][0]]
                    else:
                        new_sentence += sentence[ner_doc_final[i][1]:]

            new_sentence = new_sentence.replace(mask + "'s", mask)
            new_sentence = new_sentence.replace(mask + "'", mask)

            masked_sentences.append(new_sentence)

            """# deal with multiple masks

            s = ''
            counter_mask = 0
            match_list = []
            for match in re.finditer(mask, new_sentence):
                match_list.append([match.start(), match.end(), mask + str(counter_mask)])
                counter_mask += 1

            print("match_list", match_list)

            if counter_mask == 0:
                new_sentence1 = new_sentence
            else:
                new_sentence1 = ''
                if len(match_list) == 0:
                    new_sentence1 = new_sentence
                else:
                    new_sentence1 = new_sentence[0:match_list[0][0]]
                    num_of_ent = len(match_list)
                    for i in range(num_of_ent):
                        new_sentence1 += match_list[i][2]
                        if i != len(match_list) - 1:
                            new_sentence1 += new_sentence[match_list[i][1]: match_list[i + 1][0]]
                        else:
                            new_sentence1 += new_sentence[match_list[i][1]:]

            print("new_sentence1", new_sentence1)

            masked_sentences.append(new_sentence1)"""

        indices = list(set(indices))

    return masked_sentences, indices


def mask_multiple_words_version1(mask,new_sentence):
    # deal with multiple masks
    s = ''
    counter_mask = 0
    match_list = []  # how many times "mask" appears in the final sentence
    for match in re.finditer(mask, new_sentence):
        match_list.append([match.start(), match.end(), mask + str(counter_mask)])
        counter_mask += 1

    if counter_mask == 0:
        new_sentence1 = new_sentence
    else:
        new_sentence1 = ''
        if len(match_list) == 0:
            new_sentence1 = new_sentence
        else:
            new_sentence1 = new_sentence[0:match_list[0][0]]
            num_of_ent = len(match_list)
            for i in range(num_of_ent):
                new_sentence1 += match_list[i][2]
                if i != len(match_list) - 1:
                    new_sentence1 += new_sentence[match_list[i][1]: match_list[i + 1][0]]
                else:
                    new_sentence1 += new_sentence[match_list[i][1]:]
    return new_sentence1


def mask_multiple_words_version2(mask,new_sentence):
    # deal with multiple masks
    s = ''
    #counter_mask = 0
    match_list = []  # how many times "mask" appears in the final sentence
    for match in re.finditer(mask, new_sentence):
        match_list.append([match.start(), match.end(), mask])
        #counter_mask += 1

    counter_mask = 0
    match_list_new = []
    for i in range(len(match_list)):
        if i != 0:
            if match_list[i][0] == match_list[i-1][1] +1 :
                counter_mask +=1
            else:
                counter_mask = 0
        match_list_new.append([match_list[i][0], match_list[i][1], mask + str(counter_mask)])

    #################################
    """if counter_mask == 0:
        new_sentence1 = new_sentence
    else:"""
    new_sentence1 = ''
    if len(match_list_new) == 0:
        new_sentence1 = new_sentence
    else:
        new_sentence1 = new_sentence[0:match_list_new[0][0]]
        num_of_ent = len(match_list_new)
        for i in range(num_of_ent):
            new_sentence1 += match_list_new[i][2]
            if i != len(match_list_new) - 1:
                new_sentence1 += new_sentence[match_list_new[i][1]: match_list_new[i + 1][0]]
            else:
                new_sentence1 += new_sentence[match_list_new[i][1]:]

    return new_sentence1



def mask_text_version1(txt_path, parsed_model, mask, mask_type=None):
    """
    :return: masked sentnces + indices of masked sentences (ones containing the mask)
    here mask is --> VERB1 VERB2...VERB3....
    """

    print("version 1") ###

    if type(txt_path) == list:
        sentences = txt_path

    else:
        sentences = load_sentences(txt_path)

    masked_sentences = []
    indices = []

    if mask_type == "pos":

        match_list = []

        for i in range(len(sentences)):

            doc = parsed_model[i]

            if not sentences[i]:
                sentence_temp = []
                masked_sentences.append(" ".join(sentence_temp))
            else:
                for sent in doc.sentences:

                    sentence_temp = []
                    for word in sent.words:
                        if word.upos == mask:
                            sentence_temp.append(mask)
                            indices.append(i)

                            word_position = word.misc
                            word_start = int(word_position.split("|")[0].split("=")[1])
                            word_end = int(word_position.split("|")[1].split("=")[1])
                            match_list.append([word_start,word_end,mask])



                        else:
                            sentence_temp.append(word.text)

                    if mask + "'" in sentence_temp:
                        sentence_temp = [mask if x == mask + "'" else x for x in sentence_temp]
                    if mask + "'s" in sentence_temp:
                        sentence_temp = [mask if x == mask + "'s" else x for x in sentence_temp]

                    new_sentence = " ".join(sentence_temp)

                    ########################SECOND TRIAL############################
                    # deal with multiple masks
                    new_sentence1 = mask_multiple_words_version1(mask,new_sentence)

                    print("new_sentence1", new_sentence1)

                    masked_sentences.append(new_sentence1)
                    ################################################################

        indices = list(set(indices))

    if mask_type == "feat":

        for i in range(len(sentences)):
            doc = parsed_model[i]
            for sent in doc.sentences:
                sentence_temp = []

                for word in sent.words:
                    if (word.feats is not None) and (
                            word.feats.split("|")[0].split("=")[0] == mask.split("_")[0]): #not working with this case anymore

                        if len(mask.split("_")) > 1:  # if we split the mask by pos types

                            upos_list.append(word.upos)

                            if mask.split("_")[
                                1] == "NOUN":  # mask only intersection of feat with NOUN

                                if word.upos == "PRON":
                                    sentence_temp.append(
                                        mask)  # Append the feature itself (value for Gender/Number etc)
                                    indices.append(i)

                            if mask.split("_")[
                                1] == "VERB":  # mask only intersection of feat with NOUN

                                if word.upos == "VERB":
                                    sentence_temp.append(
                                        mask)  # Append the feature itself (value for Gender/Number etc)
                                    indices.append(i)

                        ######

                        else:  # regular case (feat not splitted to pos types)

                            sentence_temp.append(
                                mask)  # Append the feature itself (value for Gender/Number etc)
                            indices.append(i)

                        ######

                    else:

                        sentence_temp.append(word.text)

            if mask + "'" in sentence_temp:
                sentence_temp = [mask if x == mask + "'" else x for x in sentence_temp]
            if mask + "'s" in sentence_temp:
                sentence_temp = [mask if x == mask + "'s" else x for x in sentence_temp]

            new_sentence = " ".join(sentence_temp)

            print("new_sentence",new_sentence) ##

            new_sentence1 = mask_multiple_words_version1(mask, new_sentence)

            print("new_sentence1", new_sentence1)

            masked_sentences.append(new_sentence1)


        indices = list(set(indices))

    if mask_type == "ner":

        for i in range(len(sentences)):
            sentence = sentences[i]
            doc = parsed_model[i]

            ner_doc_final = []  ##

            for ent in doc.ents:


                if ent.type == mask:

                    if ent.text.startswith("the"):

                        print("ent.text.startswith(the)")

                        ner_doc = []

                        ner_doc.append(ent.start_char + 4)
                        ner_doc.append(ent.end_char)

                        ent_text = ent.text
                        print("ent.text", ent_text)
                        ent_text = ent_text.strip("the ")
                        print("ent_text.strip(the)", ent.text)

                        ent_list = ent_text.split(" ")

                        exp = ent.type

                        for ind_ent in range(1, len(ent_list)):
                            exp = exp + " " + ent.type

                        ner_doc.append(exp)


                        indices.append(i)  ##here or tab?

                        ner_doc_final.append(ner_doc)



                    else:
                        ner_doc = []

                        ner_doc.append(ent.start_char)
                        ner_doc.append(ent.end_char)

                        ent_text = ent.text
                        print("ent.text", ent.text)
                        print("ent.start_char", ent.start_char)
                        print("ent.end_char", ent.end_char)

                        ent_list = ent_text.split(" ")
                        ##print("ent_list", ent_list)

                        exp = ent.type

                        for ind_ent in range(1, len(ent_list)):
                            exp = exp + " " + ent.type

                        ner_doc.append(exp)


                        indices.append(i)  ##here or tab?

                        ner_doc_final.append(ner_doc)
                        #####

            #####

            print("ner_doc_final", ner_doc_final)

            new_sentence = ''
            if len(ner_doc_final) == 0:
                new_sentence = sentence
            else:
                new_sentence = sentence[0:ner_doc_final[0][0]]
                num_of_ent = len(ner_doc_final)
                for i in range(num_of_ent):
                    new_sentence += ner_doc_final[i][2]
                    if i != len(ner_doc_final) - 1:
                        new_sentence += sentence[ner_doc_final[i][1]: ner_doc_final[i + 1][0]]
                    else:
                        new_sentence += sentence[ner_doc_final[i][1]:]


            new_sentence = new_sentence.replace(mask + "'s", mask)
            new_sentence = new_sentence.replace(mask + "'", mask)

            # deal with multiple masks
            new_sentence1 = mask_multiple_words_version1(mask,new_sentence)

            print("new_sentence1", new_sentence1)

            # new_sentence1 is with mask+number
            masked_sentences.append(new_sentence1)

        indices = list(set(indices))


    return masked_sentences, indices

def mask_text_version2(txt_path, parsed_model, mask, mask_type=None):
    """
    :return: masked sentnces + indices of masked sentences (ones containing the mask)
    here mask is --> VERB1 VERB2...VERB3....
    """

    print("version 2")

    if type(txt_path) == list:
        sentences = txt_path

    else:
        sentences = load_sentences(txt_path)

    masked_sentences = []
    indices = []

    if mask_type == "pos":

        match_list = []

        for i in range(len(sentences)):

            doc = parsed_model[i]

            if not sentences[i]:
                sentence_temp = []
                masked_sentences.append(" ".join(sentence_temp))
            else:
                for sent in doc.sentences:

                    sentence_temp = []
                    for word in sent.words:
                        if word.upos == mask:
                            sentence_temp.append(mask)
                            indices.append(i)

                            word_position = word.misc
                            word_start = int(word_position.split("|")[0].split("=")[1])
                            word_end = int(word_position.split("|")[1].split("=")[1])
                            match_list.append([word_start,word_end,mask])



                        else:
                            sentence_temp.append(word.text)

                    if mask + "'" in sentence_temp:
                        sentence_temp = [mask if x == mask + "'" else x for x in sentence_temp]
                    if mask + "'s" in sentence_temp:
                        sentence_temp = [mask if x == mask + "'s" else x for x in sentence_temp]

                    new_sentence = " ".join(sentence_temp)

                    ########################SECOND TRIAL############################
                    # deal with multiple masks
                    new_sentence1 = mask_multiple_words_version2(mask,new_sentence)

                    print("new_sentence1", new_sentence1)

                    masked_sentences.append(new_sentence1)
                    ################################################################

        indices = list(set(indices))

    if mask_type == "feat":

        for i in range(len(sentences)):
            doc = parsed_model[i]
            for sent in doc.sentences:
                sentence_temp = []

                for word in sent.words:

                    if (word.feats is not None) and (
                            word.feats.split("|")[0].split("=")[0] == mask.split("_")[0]): #not working with this case anymore

                        if len(mask.split("_")) > 1:  # if we split the mask by pos types

                            upos_list.append(word.upos)

                            if mask.split("_")[
                                1] == "NOUN":  # mask only intersection of feat with NOUN

                                if word.upos == "PRON":
                                    sentence_temp.append(
                                        mask)  # Append the feature itself (value for Gender/Number etc)
                                    indices.append(i)

                            if mask.split("_")[
                                1] == "VERB":  # mask only intersection of feat with NOUN

                                if word.upos == "VERB":
                                    sentence_temp.append(
                                        mask)  # Append the feature itself (value for Gender/Number etc)
                                    indices.append(i)

                        ######

                        else:  # regular case (feat not splitted to pos types)

                            sentence_temp.append(
                                mask)  # Append the feature itself (value for Gender/Number etc)
                            indices.append(i)

                        ######

                    else:

                        sentence_temp.append(word.text)

            if mask + "'" in sentence_temp:
                sentence_temp = [mask if x == mask + "'" else x for x in sentence_temp]
            if mask + "'s" in sentence_temp:
                sentence_temp = [mask if x == mask + "'s" else x for x in sentence_temp]

            new_sentence = " ".join(sentence_temp)

            print("new_sentence",new_sentence) ##

            new_sentence1 = mask_multiple_words_version2(mask, new_sentence)

            print("new_sentence1", new_sentence1)

            masked_sentences.append(new_sentence1)


        indices = list(set(indices))

    if mask_type == "ner":

        for i in range(len(sentences)):
            sentence = sentences[i]
            doc = parsed_model[i]

            ner_doc_final = []  ##

            for ent in doc.ents:


                if ent.type == mask:

                    if ent.text.startswith("the"):

                        print("ent.text.startswith(the)")

                        ner_doc = []

                        ner_doc.append(ent.start_char + 4)
                        ner_doc.append(ent.end_char)

                        ent_text = ent.text
                        print("ent.text", ent_text)
                        ent_text = ent_text.strip("the ")
                        print("ent_text.strip(the)", ent.text)

                        ent_list = ent_text.split(" ")

                        exp = ent.type

                        for ind_ent in range(1, len(ent_list)):
                            exp = exp + " " + ent.type

                        ner_doc.append(exp)


                        indices.append(i)  ##here or tab?

                        ner_doc_final.append(ner_doc)



                    else:
                        ner_doc = []

                        ner_doc.append(ent.start_char)
                        ner_doc.append(ent.end_char)

                        ent_text = ent.text
                        print("ent.text", ent.text)
                        print("ent.start_char", ent.start_char)
                        print("ent.end_char", ent.end_char)

                        ent_list = ent_text.split(" ")
                        ##print("ent_list", ent_list)

                        exp = ent.type

                        for ind_ent in range(1, len(ent_list)):
                            exp = exp + " " + ent.type

                        ner_doc.append(exp)


                        indices.append(i)  ##here or tab?

                        ner_doc_final.append(ner_doc)
                        #####

            #####

            print("ner_doc_final", ner_doc_final)

            new_sentence = ''
            if len(ner_doc_final) == 0:
                new_sentence = sentence
            else:
                new_sentence = sentence[0:ner_doc_final[0][0]]
                num_of_ent = len(ner_doc_final)
                for i in range(num_of_ent):
                    new_sentence += ner_doc_final[i][2]
                    if i != len(ner_doc_final) - 1:
                        new_sentence += sentence[ner_doc_final[i][1]: ner_doc_final[i + 1][0]]
                    else:
                        new_sentence += sentence[ner_doc_final[i][1]:]


            new_sentence = new_sentence.replace(mask + "'s", mask)
            new_sentence = new_sentence.replace(mask + "'", mask)

            # deal with multiple masks
            new_sentence1 = mask_multiple_words_version2(mask,new_sentence)

            print("new_sentence1", new_sentence1)

            # new_sentence1 is with mask+number
            masked_sentences.append(new_sentence1)

        indices = list(set(indices))

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

    ###added this to debug negative score issue -- DELETE LATER###

    DIR_OUT_SIGN = "/cs/snapless/oabend/tailin/MT/NEW/outputs/negative_scores/dicts/"
    ##############################################

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
                temp_filename = "temp"

                ###############################################################

            save_pickle(DIR_OUT + temp_filename + "_sentence_bleu_scores.txt", bleu)
        else:
            bleu = []
            for i in indices:
                score += sentence_bleu([reference[i].strip().split()], candidate[i].strip().split())
                bleu.append(
                    sentence_bleu([reference[i].strip().split()], candidate[i].strip().split()))

                ###############################################################

    # score /= len(reference) FIXED THIS (15_10_21)
    score /= len(indices)

    ###############################################################
    print("^" * 20)
    print("I'm inside score_sentence_bleu function")
    print("len(indices), len(bleu): ", len(indices), len(bleu))
    print("^" * 20)

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
        current_bleu_score, _ = score_sentence_bleu(reference, candidate1, DIR_OUT=None, indices=None, save_score=False)

        list_bleu_per_candidate.append(current_bleu_score)

    max_bleu = max(list_bleu_per_candidate)  # the max bleu score
    index_max = list_bleu_per_candidate.index(
        max(list_bleu_per_candidate))  # index of candidate with max bleu score

    # print("max bleu score is ", max_bleu)

    max_bleu_dict = {}
    max_bleu_dict["max_bleu"] = max_bleu
    max_bleu_dict["index_max"] = index_max

    return max_bleu_dict


def parse_ref_candidate(DIR_OUT, trg_lang, mask_type, reference, current_candidate, temp_filename):  # DELETE LATER
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


def score_sentence_rouge(reference_path, candidate_path, DIR_OUT=None, indices=None, save_score=False):
    """
    reference_path = path to masked (or not masked) references file
    candidate_path = path to masked (or not masked) translations file
    indices = list of indices of rows containing the mask (intersection/union)
    return: (sentence) rougeL percision score
    """

    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

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

    # score /= len(reference) FIXED THIS (15_10_21)
    score /= len(indices)

    return score, score_list


######################################################################################################################

# full process (seperately for pos,ner; later I will join them)

def compute_scores(ref_input, candidate_input, DIR_OUT, parsed_model_ref, parsed_model_candidate, mask_list_path,
                   mask_type=None, score_type=None, mask_text_version = None):
    """
    :param score_type: "bleu"/"rouge" (if None --> score_type == 'bleu')
    """
    # note: parsed_model_ref & parsed_model_candidate are dictionaries, they are also saved they can be loaded with pickle
    # note: indices are not a parameter of this function since they are computes inside the function

    if mask_text_version is None:
        mask_text = mask_text_version0
    if mask_text_version == "all":
        mask_text = mask_text_version1
    if mask_text_version == "sets":
        mask_text = mask_text_version2


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

    bleu_not_masked, bleu_not_masked_list = score_function(ref_input, candidate_input,
                                                           DIR_OUT, indices=None,
                                                           save_score=True)

    print("len(bleu_not_masked_list) -- over ALL sentences", len(bleu_not_masked_list))

    print("**score_type**",score_type)

    dict_results[score_type + "_not_masked"] = bleu_not_masked  # ORIGINAL #this is not over the indices, it's the original bleu score of the file! it can be used to choose the file with the best bleu for example

    for mask in mask_list:

        print("mask", mask)

        masked_ref, indices_ref = mask_text(ref_input, parsed_model_ref, mask, mask_type)
        masked_candidate, indices_candidate = mask_text(candidate_input, parsed_model_candidate,
                                                        mask, mask_type)

        ################################################################################################################################

        indices = indices_intersection(indices_ref, indices_candidate)

        print("len(indices)", len(indices))

        #####DELETE LATER#####
        print("#"*50)
        print(mask)
        for ind1 in indices:
            print("reference: " + ref_input[ind1])
            print("reference_masked: " + masked_ref[ind1])
            print("*"*10)
            print("candidate: " + candidate_input[ind1])
            print("candidate_masked: " + masked_candidate[ind1])

        print("#" * 50)
        ######################
        for ind1 in indices:
            with open(DIR_OUT + mask_type + "_" + score_type + "_check.txt", "w") as outfile1:
                outfile1.write("reference_masked: " + masked_ref[ind1] + "\n")
                outfile1.write("reference: " + ref_input[ind1] + "\n")
                outfile1.write("candidate_masked: " + masked_candidate[ind1] + "\n")
                outfile1.write("candidate: " + candidate_input[ind1] + "\n")
                outfile1.write("*" * 100 + "\n")
        ######################

        # bleu

        if len(indices) == 0:

            total_bleu = "NA"
        else:
            # score_masked, _ = score_function(masked_ref, masked_candidate, indices,save_score=False)
            score_masked, bleu_masked_list = score_function(masked_ref, masked_candidate, DIR_OUT, indices,
                                                            save_score=False)  # THERE IS A BUG HERE --> for some reason the function doesn't read the indices correctly and treats them as None
            ####added this here#####
            score_not_masked, bleu_not_masked_list = score_function(ref_input, candidate_input,
                                                                    DIR_OUT, indices,
                                                                    save_score=False)  # now this is the bleu score only over the indices

            print("len(bleu_not_masked_list)", len(bleu_not_masked_list))
            print("len(bleu_masked_list)", len(bleu_masked_list))

            total_bleu = score_masked - score_not_masked  # ORIGINAL

            print("score_masked [from score function]", score_masked)
            print("score_not_masked [from score function]", score_not_masked)
            print("total bleu [score masked - score_not_masked]", total_bleu)

            ##sanity check##
            score_masked_check = 0
            score_not_masked_check = 0

            for i in range(len(bleu_not_masked_list)):
                score_not_masked_check += bleu_not_masked_list[i]
            score_not_masked_check /= len(bleu_not_masked_list)
            for i in range(len(bleu_masked_list)):
                score_masked_check += bleu_masked_list[i]
            score_masked_check /= len(bleu_masked_list)
            print("compute score manually with the sum")
            print("score_masked", score_masked_check)
            print("score_not_masked", score_not_masked_check)

            print("FINAL CHECK")
            a = 0
            for i in range(len(bleu_masked_list)):
                a += bleu_masked_list[i]
            a /= len(bleu_masked_list)
            print("score_masked [sum ONLY over indices]", a)
            print("total bleu [sum ONLY over indices]", a - score_not_masked_check)
            print("total bleu [sum over WRONG indices]", total_bleu)

            print("*" * 20)
            counter_neg_score = 0.
            for i in range(len(bleu_not_masked_list)):
                if bleu_masked_list[i] < bleu_not_masked_list[i]:
                    counter_neg_score += 1

            print("counter_neg_score", counter_neg_score)
            ################

        dict_bleu[mask] = total_bleu

        # hallucination
        add, miss, hit = hallucination_score(masked_ref, masked_candidate, mask, indices)
        dict_hal[mask] = [add, miss, hit]

    dict_results[score_type] = dict_bleu
    dict_results["hallucination"] = dict_hal

    return dict_results


def parse_file(DIR_OUT, mask_type, input, temp_filename, trg_lang="en"):
    """
    :param DIR_OUT: where to save parsed dict (will also look for parsed dict there)
    :param mask_type: mask_type (to save in the name of the file) + parse the model accordingly
    :param input: reference/candidate (txt)
    :param temp_filename: temporary filename to use in the name of the saved parsed dict
    :param trg_lang: the stanza parser needs this info (default is "en")
    :return: parsed dictionary + save the parsed dictionary to DIR_OUT if it doesn't already exist
    """

    print("temp_filename", temp_filename)
    print("DIR_OUT", DIR_OUT)
    print("mask_type", mask_type)

    CHECK_FOLDER_REF = os.path.isfile(DIR_OUT + temp_filename + "_dict_parse_" + mask_type + ".txt")

    if not (CHECK_FOLDER_REF):

        dict_parse = preprocess(input, DIR_OUT,
                                temp_filename + "_dict_parse",
                                lang=trg_lang,
                                model_type=mask_type, mask_type=mask_type, save=True)

    else:

        dict_parse = load_pickle(
            DIR_OUT + temp_filename + "_dict_parse_" + mask_type + ".txt")

    return dict_parse


def generate_filename():
    # datetime object containing current date and time
    now = datetime.now()
    dt_string = now.strftime("%d:%m:%Y:%H:%M:%S")
    return dt_string


######################################################################################################################
# main function


def run_main_old(DIR_OUT, mask_list_path, ref_input, candidates_input, mask_type, trg_lang="en",
                 max_bleu_dict_path=None, run_all=False, score_type=None,mask_text_version = None):
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

            temp_filename = generate_filename()  # generate temporary filename, to save parsed dictionaries

            dict_parse_ref_mask, dict_parse_candidate_mask = parse_ref_candidate(DIR_OUT, trg_lang,
                                                                                 mask_type,
                                                                                 reference,
                                                                                 current_candidate,
                                                                                 temp_filename)  # HERE

            ###CONTINUE###

            dict_results = compute_scores(reference, current_candidate, DIR_OUT,
                                          dict_parse_ref_mask,
                                          dict_parse_candidate_mask,
                                          mask_list_path, mask_type,score_type,mask_text_version)

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


        dict_results = compute_scores(reference, current_candidate, DIR_OUT, dict_parse_ref_mask,
                                      dict_parse_candidate_mask,
                                      mask_list_path, mask_type, score_type, mask_text_version)

        return dict_results


def run_main(DIR_OUT, mask_list_path, ref_input, candidates_input, mask_type, trg_lang="en",
             max_bleu_dict_path=None, run_all=False, score_type=None,mask_text_version=None):
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
    #NOTE: ONLY WORKS FOR BLEU (NOT ROUGE)

    reference, candidates = load_ref_candidates(ref_input, candidates_input)

    #####################$$$$$$$$$$$$$$$$$$%%%%%%%%%%%%%%%%^^^^^^^^^^^^^^^^^^

    if score_type == "bleu" or score_type is None:

        if run_all is True:

            # run on all candidate files, output a list of results dictionary (i-th element is the results dict for the i-th candidate)

            list_results_dictionary = []

            ######added this for Gal###########
            if type(ref_input) == str:
                temp_filename_ref = ref_input.strip(".txt")
                temp_filename_ref = temp_filename_ref.split("/")
                temp_filename_ref = temp_filename_ref[len(temp_filename_ref) - 1]

            else:
                temp_filename_ref = "reference_" + generate_filename()  # generate temporary filename, to save parsed dictionaries
            ###################################

            for ind_candidate in range(len(candidates)):

                print("len(candidates)", len(candidates))

                current_candidate = candidates[ind_candidate]

                print("current_candidate", current_candidate)

                ######added this for Gal###########

                if type(candidates_input[ind_candidate]) == str:

                    print("candidates_input[ind_candidate]", candidates_input[ind_candidate])

                    temp_filename_candidate = candidates_input[ind_candidate].strip(".txt")
                    temp_filename_candidate = temp_filename_candidate.split("/")
                    temp_filename_candidate = temp_filename_candidate[len(temp_filename_candidate) - 1]

                else:
                    temp_filename_candidate = "candidate_" + generate_filename()  # generate temporary filename, to save parsed dictionaries
                ###################################

                DIR_OUT_PARSE = DIR_OUT + "parsed_files/"
                make_folder(DIR_OUT_PARSE)

                dict_parse_ref_mask = parse_file(DIR_OUT_PARSE, mask_type, reference, temp_filename_ref, trg_lang)
                dict_parse_candidate_mask = parse_file(DIR_OUT_PARSE, mask_type, current_candidate,
                                                       temp_filename_candidate, trg_lang)

                ###CONTINUE###

                dict_results = compute_scores(reference, current_candidate, DIR_OUT,
                                              dict_parse_ref_mask,
                                              dict_parse_candidate_mask,
                                              mask_list_path, mask_type,score_type,mask_text_version)

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

            ######added this for Gal --- FIX THIS PART!!!!!!!!###########
            if type(ref_input) == str:
                temp_filename_ref = ref_input.strip(".txt")
                temp_filename_ref = temp_filename_ref.split("/")
                temp_filename_ref = temp_filename_ref[len(temp_filename_ref) - 1]

            else:
                temp_filename_ref = "reference_" + generate_filename()  # generate temporary filename, to save parsed dictionaries

            if type(current_candidate) == str:

                temp_filename_candidate = current_candidate.strip(".txt")
                temp_filename_candidate = temp_filename_candidate.split("/")
                temp_filename_candidate = temp_filename_candidate[len(temp_filename_ref) - 1]

            else:
                temp_filename_candidate = "candidate_" + generate_filename()  # generate temporary filename, to save parsed dictionaries

            DIR_OUT_PARSE = DIR_OUT + "parsed_files/"
            make_folder(DIR_OUT_PARSE)

            # print("DIR_OUT_PARSE",DIR_OUT_PARSE)

            dict_parse_ref_mask = parse_file(DIR_OUT_PARSE, mask_type, reference, temp_filename_ref, trg_lang)
            dict_parse_candidate_mask = parse_file(DIR_OUT_PARSE, mask_type, current_candidate, temp_filename_candidate,
                                                   trg_lang)

            ######################################

            # now we can use the parsed files instead of parsing at each iteration
            # compute scores (for each mask_type)

            dict_results = compute_scores(reference, current_candidate, DIR_OUT, dict_parse_ref_mask,
                                          dict_parse_candidate_mask,
                                          mask_list_path, mask_type, score_type,mask_text_version)

    else:

        if run_all is True:

            # run on all candidate files, output a list of results dictionary (i-th element is the results dict for the i-th candidate)

            list_results_dictionary = []

            for ind_candidate in range(len(candidates)):
                current_candidate = candidates[ind_candidate]

                temp_filename = generate_filename()  # generate temporary filename, to save parsed dictionaries

                dict_parse_ref_mask, dict_parse_candidate_mask = parse_ref_candidate(DIR_OUT, trg_lang,
                                                                                     mask_type,
                                                                                     reference,
                                                                                     current_candidate,
                                                                                     temp_filename)  # HERE

                ###CONTINUE###

                dict_results = compute_scores(reference, current_candidate, DIR_OUT,
                                              dict_parse_ref_mask,
                                              dict_parse_candidate_mask,
                                              mask_list_path, mask_type,score_type,mask_text_version)

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

            dict_results = compute_scores(reference, current_candidate, DIR_OUT, dict_parse_ref_mask,
                                          dict_parse_candidate_mask,
                                          mask_list_path, mask_type, score_type,mask_text_version)

        return dict_results


######################################################################################################################
######################################################################################################################
######################################################################################################################
# main
#note for gal: if you want to use the existing parsed file, in "run_main" function you need to supply "DIR_OUT" which contains the parsed ref_input and parsed candidate_input

# ARGS
DIR_OUT = "/cs/snapless/oabend/tailin/MT/NEW/outputs/20.10.21/"

ref_input = "/cs/snapless/oabend/tailin/MT/NEW/codes/trial_data/181021/ref_ner_neg.txt"
# ref_input = "/cs/snapless/oabend/borgr/SSMT/data/submissions/wmt19/plain/references/newstest2019-deen-ref.en"


candidates_inputs = [["/cs/snapless/oabend/tailin/MT/NEW/codes/trial_data/181021/can_ner_neg.txt"]]
# candidates_inputs = [["/cs/snapless/oabend/borgr/SSMT/data/submissions/wmt19/plain/system-outputs/newstest2019/de-en/newstest2019.Facebook_FAIR.6750.de-en"],["/cs/snapless/oabend/borgr/SSMT/data/submissions/wmt19/plain/system-outputs/newstest2019/de-en/newstest2019.online-A.0.de-en"],["/cs/snapless/oabend/borgr/SSMT/data/submissions/wmt19/plain/system-outputs/newstest2019/de-en/newstest2019.UCAM.6461.de-en"],["/cs/snapless/oabend/borgr/SSMT/data/submissions/wmt19/plain/system-outputs/newstest2019/de-en/newstest2019.uedin.6749.de-en"]]
# candidates_input = candidates_inputs[0]

mask_types = ["pos","ner"]

for mask_type in mask_types:

    print("mask_type",mask_type)

    for candidates_input in candidates_inputs:

        print("mask_type", mask_type)

        mask_list_path = "/cs/snapless/oabend/tailin/MT/NEW/codes/trial_data/" + mask_type + "_mask_list.txt"

        result = run_main(DIR_OUT, mask_list_path, ref_input, candidates_input, mask_type, trg_lang="en",
                              max_bleu_dict_path=None, run_all=True, score_type="bleu", mask_text_version=None)

        print("result")
        print(result)

        print("%" * 100)
        print("%" * 100)

##################################################

