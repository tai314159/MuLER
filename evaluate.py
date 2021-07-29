import warnings
from concreteness_score import ConcretenessScorer
from vad_scores import ValenceScorer
from sentiment_score import SentimentScorer
import numpy as np
# from compute_score_new import score_sentence_bleu

METRICS = ['concreteness', 'sentiment', 'valence']
MASKING = []
LANGS = ['en']
# nltk.data.path.append('/cs/labs/oabend/gal.patel/virtualenvs/resources')
import nltk
nltk.download('stopwords', download_dir='/cs/labs/oabend/gal.patel/virtualenvs/mteval-venv/nltk_data')
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
    return metric2method

def eval(references, candidates, lang='en', metric_names=None, masking=None):
    if lang not in LANGS:
        raise RuntimeError(lang+ ' not supported')
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
        if len(references) != len(can):
            raise RuntimeError('candidate list ' + str(i) + ' does not match references in length')
        can_metrics = dict()
        # can_metrics['sentence_bleu'], _ = score_sentence_bleu(references, can)
        for m in metric_names:
            can_metrics[m+'_diff'] = np.mean(np.abs(metric2method[m].score_batch(can) - ref_metircs[
                m]))
        candidates_metrics.append(can_metrics)


    if len(candidates_metrics) == 1:
        return candidates_metrics[0]
    return candidates_metrics
