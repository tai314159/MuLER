import warnings
from concreteness_score import ConcretenessScorer
from vad_scores import ValenceScorer, DominanceScorer, ArousalScorer
from sentiment_score import SentimentScorer
import numpy as np
# from masker import score_sentence_bleu
from compute_scores_201021 import score_sentence_bleu
from compute_scores_201021 import run_main as run_mask
import os
import sys

# RUN_CACHE = 'run_cache'
# os.makedirs(RUN_CACHE, exist_ok=True)
METRICS = ['concreteness', 'sentiment', 'valence', 'dominance', 'arousal']
MASKING = ["pos", "ner", "feat"]
LANGS = ['en']
# nltk.data.path.append('/cs/labs/oabend/gal.patel/virtualenvs/resources')
import nltk

nltk.download('stopwords',
              download_dir='/cs/labs/oabend/gal.patel/virtualenvs/mteval-venv/nltk_data')
nltk.download('punkt', download_dir='/cs/labs/oabend/gal.patel/virtualenvs/mteval-venv/nltk_data')

def path2list(filepath, return_list_always=False):
    """
    Convert from a file path in standart form (each line is a text item) to a list of str
    If filepath is a list of paths, return a list of lists
    """
    if type(filepath) == str:
        paths = [filepath]
    else:
        paths = filepath
    txts = []
    for path in paths:
        txt_list = []
        with open(path, 'r') as txtfile:
            for line in txtfile:
                line = line.rstrip("\n")
                txt_list.append(line)
        txts.append(txt_list)
    if len(txts) == 1 and not return_list_always:
        return txts[0]
    return txts

def get_names(options, queries):
    if queries is None:
        return options
    names = []
    for query in queries:
        if query not in options:
            warnings.warn(query + ' not found')
        else:
            names.append(query)
    return names


def get_methods(lang, metric_names):
    metric2method = dict()
    for m in metric_names:
        if m == 'concreteness':
            metric2method[m] = ConcretenessScorer(lang)
        elif m == 'sentiment':
            metric2method[m] = SentimentScorer(lang)
        elif m == 'valence':
            metric2method[m] = ValenceScorer(lang)
        elif m == 'dominance':
            metric2method[m] = DominanceScorer(lang)
        elif m == 'arousal':
            metric2method[m] = ArousalScorer(lang)
    return metric2method


def tailin():
    for mask_type1 in mask_types:

        # parse dicts

        DIR_OUT_PARSE = DIR_OUT + "parsed_dicts_stanza/" + str(key) + "/" + mask_type1 + "/"

        make_folder(DIR_OUT_PARSE)

        CHECK_FOLDER_REF = os.path.isfile(
            DIR_OUT_PARSE + src_lang + "_" + trg_lang + "_dict_parse_ref_" + mask_type1 + ".txt")
        CHECK_FOLDER_CAN = os.path.isfile(
            DIR_OUT_PARSE + src_lang + "_" + trg_lang + "_dict_parse_candidate_" + mask_type1 + ".txt")

        if not (CHECK_FOLDER_REF & CHECK_FOLDER_CAN):
            print("parsing" + src_lang + "_" + trg_lang + "!")

            dict_parse_ref_mask = preprocess(reference_path, DIR_OUT_PARSE,
                                             src_lang + "_" + trg_lang + "_dict_parse_ref",
                                             lang="en",
                                             model_type=mask_type1, mask_type=mask_type1, save=True)

            dict_parse_candidate_mask = preprocess(candidate_path, DIR_OUT_PARSE,
                                                   src_lang + "_" + trg_lang + "_dict_parse_candidate",
                                                   lang="en", model_type=mask_type1,
                                                   mask_type=mask_type1,
                                                   save=True)
        else:
            dict_parse_ref_mask = load_pickle(
                DIR_OUT_PARSE + src_lang + "_" + trg_lang + "_dict_parse_ref_" + mask_type1 + ".txt")
            dict_parse_candidate_mask = load_pickle(
                DIR_OUT_PARSE + src_lang + "_" + trg_lang + "_dict_parse_candidate_" + mask_type1 + ".txt")
            ########################################################################

        # now we can use the parsed files instead of parsing at each iteration
        # compute scores (for each mask_type)
        DIR_OUT_FINAL = DIR_OUT + "final_results/"
        make_folder(DIR_OUT_FINAL)

        DIR_OUT_TEMP = DIR_OUT + "temp_results/"
        make_folder(DIR_OUT_TEMP)

        # note: parsed_model_ref & parsed_model_candidate are dictionaries, they are also saved they can be loaded with pickle
        compute_scores(reference_path, candidate_path, DIR_OUT_FINAL, DIR_OUT_TEMP,
                       dict_parse_ref_mask,  # preprocessed
                       dict_parse_candidate_mask,
                       model_type=mask_type1, lang="en",
                       mask_list_path=mask_list_dict[mask_type1],  # what to mask (ie noun)
                       mask_type=mask_type1)

        # THE END


def eval_muler(references, candidates, lang='en', metric_names=None, masking=None, cache_dir=None,
               are_paths=True, always_return_list=True, version=None):
    if lang not in LANGS:
        raise RuntimeError(lang + ' not supported')
    if (not are_paths) and len(candidates) and type(candidates[0]) != list:
        candidates = [candidates]
    if are_paths:
        ref_path = references
        can_paths = candidates
        # print('>>', type(can_paths), len(can_paths), type(can_paths[0]))
        # print('>>', type(ref_path))
        references = path2list(ref_path)
        candidates = path2list(can_paths, return_list_always=True)
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        print('YAY CACHE')
    else:
        raise RuntimeError('no cache')
    metric_names = get_names(METRICS, metric_names)
    maskings = get_names(MASKING, masking)
    # masking_res = dict()
    # for mask in maskings:
    #     masking_res[mask] = run_mask('mask_out', '20.07.21/mask_lists/'+mask+'_full_list.txt',
    #                                  references,
    #                                  candidates,
    #                                  mask, run_all=True)
    # print(masking_res[mask][0]['bleu'].keys())
    # sys.exit()

    # compute for references
    ref_metircs = {}
    metric2method = get_methods(lang, metric_names)
    for m in metric_names:
        ref_metircs[m] = metric2method[m].score_batch(references)

    candidates_metrics = []
    # discard_ids = []
    new_candidates = []
    for i, can in enumerate(candidates):
        can_metrics = dict()
        if len(references) != len(can):
            # print('found', i)
            # raise RuntimeError('candidate list ' + str(i) + ' does not match references in length')
            # discard_ids.append(i)
            continue
            # can_metrics['bleu'] = 0
            # for m in metric_names:
            #     can_metrics[m + '_diff'] = float('inf')
        else:

            new_candidates.append(can)
            can_metrics['bleu'], _ = score_sentence_bleu(references, can)
            for m in metric_names:
                can_metrics[m + '_diff'] = np.mean(
                    np.abs(metric2method[m].score_batch(can) - ref_metircs[
                        m]))

        candidates_metrics.append(can_metrics)
    mask_metrics = []
    # new_candidates = []
    # for i in range(len(candidates)):
    #     if i not in discard_ids:
    #         new_candidates.append(candidates[i])
    # print('candidates:', len(candidates))
    # print('new_candidates:', len(new_candidates))
    candidates = new_candidates
    # for i, can in enumerate(candidates):
    #     if len(can) != len(references):
    #         print('FOUND', i)

    try:
        for mask in maskings:
            # mask_list_path = RUN_CACHE + '/',
            # '20.07.21/mask_lists/' + mask +
            # '_full_list.txt',
            # ref_input = ref_path,
            # candidates_input = can_paths,
            # mask_type = mask, run_all = True, score_type = 'bleu', DIR_OUT = cache_dir
            print(cache_dir)
            res = run_mask(mask_list_path ='20.07.21/mask_lists/' + mask + '_full_list.txt',
                           ref_input = ref_path,
                           candidates_input=can_paths,
                           mask_type=mask, run_all=True, score_type='bleu', DIR_OUT=cache_dir,
                           mask_text_version=version)
            # mask_res = dict()
            # for k in res:
            #     mask_res[mask.upper() + '_' + k] = res[k]
            # mask_metrics.append(mask_res)
            for i, can in enumerate(candidates_metrics):
                mask_res = res[i]
                for score in ['bleu', 'hallucination']:
                    for k in mask_res[score]:
                        can[mask.upper() + '_' + k + '_' + score] = mask_res[score][k]
    except Exception as e:
        print('FAILURE OCCURED')
        print(e)
    if (not always_return_list) and len(candidates_metrics) == 1:
        return candidates_metrics[0]
    return candidates_metrics
