import warnings
from concreteness_score import ConcretenessScorer
from vad_scores import ValenceScorer, DominanceScorer, ArousalScorer
from sentiment_score import SentimentScorer
import numpy as np
from masker import score_sentence_bleu

METRICS = ['concreteness', 'sentiment', 'valence', 'dominance', 'arousal']
MASKING = ["pos", "ner", "feat"]
LANGS = ['en']
# nltk.data.path.append('/cs/labs/oabend/gal.patel/virtualenvs/resources')
import nltk

nltk.download('stopwords',
              download_dir='/cs/labs/oabend/gal.patel/virtualenvs/mteval-venv/nltk_data')
nltk.download('punkt', download_dir='/cs/labs/oabend/gal.patel/virtualenvs/mteval-venv/nltk_data')


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
                       mask_list_path=mask_list_dict[mask_type1], # what to mask (ie noun)
                       mask_type=mask_type1)

        # THE END


def eval(references, candidates, lang='en', metric_names=None, masking=None):
    if lang not in LANGS:
        raise RuntimeError(lang + ' not supported')
    if len(candidates) and type(candidates[0]) != list:
        candidates = [candidates]

    metric_names = get_names(METRICS, metric_names)
    maskings = get_names(MASKING, masking)

    # compute for references
    ref_metircs = {}
    metric2method = get_methods(lang, metric_names)
    for m in metric_names:
        ref_metircs[m] = metric2method[m].score_batch(references)

    candidates_metrics = []
    for i, can in enumerate(candidates):
        can_metrics = dict()
        if len(references) != len(can):
            # raise RuntimeError('candidate list ' + str(i) + ' does not match references in length')
            can_metrics['bleu'] = 0
            for m in metric_names:
                can_metrics[m + '_diff'] = float('inf')
        else:
            can_metrics['bleu'], _ = score_sentence_bleu(references, can)
            for m in metric_names:
                can_metrics[m + '_diff'] = np.mean(
                    np.abs(metric2method[m].score_batch(can) - ref_metircs[
                        m]))
        candidates_metrics.append(can_metrics)

    if len(candidates_metrics) == 1:
        return candidates_metrics[0]
    return candidates_metrics
