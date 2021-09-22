import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from functools import reduce

MASK_LIST = '20.07.21/mask_lists'
MASK_TYPES = ['pos', 'ner', 'feat']

sns.set(font_scale=1)
# sns.set_palette('colorblind', 5)


def get_data(data_path):
    data = pd.read_csv(data_path)
    return data


def get_type_list(type_name):
    with open(os.path.join(MASK_LIST, type_name.lower() + '_full_list.txt'), 'r') as f:
        name_list = []
        for line in f:
            name_list.append(line.strip())
        return name_list


def get_column_names(score_name, column_names):
    prefix = set([x.split('_')[0] for x in column_names])
    if score_name.upper() + '_bleu' in column_names:  # e.g. VERB_bleu
        return score_name, score_name.upper() + '_bleu'
    if score_name.lower() + '_diff' in column_names:  # e.g. sentiment_diff
        return score_name, score_name.lower() + '_diff'
    if score_name.lower() == 'masking':
        # need total of each prefix
        names, cols = [], []
        # mask_names = [get_type_list(x) for x in MASK_TYPES]
        for mask_type in MASK_TYPES:
            names.append(mask_type.upper())
            cols.append(list(map(lambda x: x + '_bleu', get_type_list(mask_type))))
            # cols.append(p + '_bleu')
        return names, cols
    if score_name.lower() in ['feat', 'ner', 'pos']:
        # need total of each mask type
        names, cols = [], []
        for p in get_type_list(score_name.lower()):
            names.append(p)
            cols.append(p + '_bleu')
        return names, cols
    if score_name == 'scorer':
        names = ['concreteness', 'sentiment', 'valence']
        cols = list(map(lambda x: x + '_diff', names))
        return names, cols
    # if score_name == 'all':


def score2dir(score_name):
    score_name = score_name.lower()
    score_name = score_name.replace(' ', '')
    return score_name

def get_lang(lang, data, lang_dir='src'):
    if type(lang) == str:
        return [lang]
    elif type(lang) == list:
        return lang
    else:
        return list(np.unique(data[lang_dir]))

def score_by_year(data_path, score_name, src=None, trg='en',
                  save_dir='analysis/score_by_year',
                  best_bleu=False):
    save_dir = os.path.join(save_dir, score2dir(score_name))
    os.makedirs(save_dir, exist_ok=True)
    all_data = get_data(data_path)
    for src_lang in get_lang(src, all_data, 'src'):
        for trg_lang in get_lang(trg, all_data, 'trg'):
            data = all_data[all_data['src'] == src_lang]
            data = data[data['trg'] == trg_lang]


            years = np.unique(data['year'])
            res_score = defaultdict(list)
            res_year = defaultdict(list)
            names, score_columns = get_column_names(score_name, data.columns)
            for year in years:
                data_year = data[data['year'] == year]
                if data_year.empty:
                    continue

                if best_bleu:
                    data_year = data_year.loc[data_year['bleu'].idxmax()]  # todo really names 'bleu'?
                for name, col_name in zip(names, score_columns):
                    res_year[name].append(year)
                    if type(col_name) is list:  # the score is a sum of columns
                        total = 0
                        for c in col_name:
                            total += data_year.loc[data_year[c].idxmax()][c]
                        res_score[name].append(total)
                    else:
                        res_score[name].append(data_year.loc[data_year[col_name].idxmax()][col_name])
            for name in names:
                if len(res_year[name]) == 1:
                    plt.scatter(res_year[name], np.array(res_score[name]), label=name)
                elif len(res_year[name]) > 1:
                    plt.plot(res_year[name], np.array(res_score[name]), label=name)
            plt.xlabel('year')
            plt.ylabel('score')
            plt.suptitle(score_name)
            plt.title(src_lang + trg_lang)
            plt.legend(loc=0)
            by = 'best_bleu' if best_bleu else 'best_score'
            plt.savefig(os.path.join(save_dir, src_lang + trg_lang + '_' + by + '.png'),
                        bbox_inches='tight')
            plt.clf()
            plt.close()


if __name__ == '__main__':
    score_by_year(data_path='scores.csv', score_name='scorer')
