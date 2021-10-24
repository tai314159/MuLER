import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from functools import reduce
import pickle
from scipy.stats import pearsonr
MASK_LIST = '20.07.21/mask_lists'
MASK_TYPES = ['pos', 'ner', 'feat']
DISCARD = {'x', 'intj'}

sns.set(font_scale=1)


# sns.set_palette('colorblind', 5)


def arrange_data(data_path='20.07.21/results_dictionaries'):
    df = pd.DataFrame(columns=['year', 'src', 'trg', 'bleu', 'feat', 'ner', 'pos'])
    res = dict()
    for file in os.listdir(data_path):
        if 'sentence' in file:
            continue
        with open(os.path.join(data_path, file), 'rb') as f:
            data = pickle.load(f)
            details = file.split('-')
            year = details[0][-4:]
            src = details[1][:2]
            trg = details[1][2:]
            key = year, src, trg
            print('key', key)
            if key not in res:
                res[key] = {'bleu': float(data['bleu_not_masked'])}
            print(data['bleu'].values())
            total = 0.
            for k in data['bleu']:
                if '_' not in k and type(data['bleu'][k]) == float:
                    # if type(data['bleu'][k])==str:
                    #     print('STR', data['bleu'][k])
                    total += data['bleu'][k]
            res[key][details[2].split('_')[1]] = total
    for year, src, trg in res.keys():
        print('->', year, src, trg)
        val = res[year, src, trg]
        df = df.append({'year': year, 'src': src, 'trg': trg,
                        'bleu': val['bleu'],
                        'feat_bleu': val['feat'], 'ner_bleu': val['ner'], 'pos_bleu': val['pos']},
                       ignore_index=True)
    return df


def get_data(data_path):
    # return arrange_data()
    data = pd.read_csv(data_path)
    clean_data = pd.DataFrame(columns=data.columns)
    for i, row in data.iterrows():
        if not 'human' in row['submission'].lower():
            clean_data = clean_data.append(row, ignore_index=True)
    # add total columns
    # for mtype in MASK_TYPES:
    #     columns = filter(lambda x: x.startswith(mtype.upper()) and 'total' not in x and
    #                                x.endswith('bleu') and x.split('_')[1] not in DISCARD, data.columns)
    #     # print(type(data[list(columns)[0]]), data[list(columns)[0]].dtype)
    #     total = np.zeros(len(data))
    #     for c in columns:
    #         # print(type(data[c]), data[c].dtype, c, data[c])
    #         total += data[c]
    #     # total = reduce(lambda x, y: np.array(data[x]) + np.array(data[y]), columns)
    #     data[mtype.upper() + '_total_bleu'] = total
    #     # print(total)
    return clean_data


def get_type_list(type_name):
    with open(os.path.join(MASK_LIST, type_name.lower() + '_full_list.txt'), 'r') as f:
        name_list = []
        for line in f:
            name_list.append(line.strip())
        return name_list


def get_column_names(score_name, column_names):
    prefix = set([x.split('_')[0] for x in column_names])
    lower_names = list(map(lambda x: x.lower(), column_names))
    if score_name == 'bleu':
        return 'bleu', 'bleu'
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
            # cols.append(list(map(lambda x: x + '_bleu', get_type_list(mask_type))))
            cols.append(mask_type.upper() + '_total_bleu')
        return names, cols
    if score_name.lower() in ['feat', 'ner', 'pos']:
        # need total of each mask type
        # return score_name, score_name.upper()+'_total_bleu'
        names, cols = [], []
        # for p in get_type_list(score_name.lower()):
        #     names.append(p)
        #     cols.append(p + '_bleu')
        # return names, cols
        for c in column_names:
            if c.lower().startswith(score_name.lower()) and 'total' not in c and 'bleu' in c:
                n = c.split('_')[1]
                if n in DISCARD:
                    continue
                names.append(n)
                cols.append(c)
        return names, cols
    if score_name == 'scorer':
        names = ['concreteness', 'sentiment', 'valence', 'dominance', 'arousal']
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



def lang_by_year(data_path, score_name, src=None, trg='en',
                 save_dir='analysis/lang_by_year',
                 best_bleu=False):
    save_dir = os.path.join(save_dir, score2dir(score_name))
    os.makedirs(save_dir, exist_ok=True)
    all_data = get_data(data_path)
    names, score_columns = get_column_names(score_name, all_data.columns)

    for name, col_name in zip(names, score_columns):
        res_score = defaultdict(list)
        res_year = defaultdict(list)
        min_year = None
        max_year = None
        for src_lang in get_lang(src, all_data, 'src'):
            for trg_lang in get_lang(trg, all_data, 'trg'):
                data = all_data[all_data['src'] == src_lang]
                data = data[data['trg'] == trg_lang]
                years = np.unique(data['year'])
                if 'wmt20' in years and years[-1] != 'wmt20':
                    print('YEARS->', years)
                    raise RuntimeError('years unsorted '+ str(years))


                for year in years:
                    data_year = data[data['year'] == year]
                    # data_year = data_year['human' not in data_year['submission'].lower()]
                    if data_year.empty or data_year[col_name].isnull().values.any():
                        continue
                    num_year = int(year[-2:])
                    if min_year is None:
                        min_year = num_year
                        max_year = num_year
                    else:
                        min_year = min(min_year, num_year)
                        max_year = max(max_year, num_year)

                    print('year:', year, 'lang:', src_lang)
                    res_year[src_lang, trg_lang].append(num_year)
                    if type(col_name) is list:  # the score is a sum of columns
                        total = 0
                        for c in col_name:
                            total += data_year.loc[data_year[c].idxmin()][c]
                        res_score[src_lang, trg_lang].append(total)
                        raise RuntimeError('why still list?')
                    else:
                        if best_bleu:
                            # data_year = data_year.loc[
                            #     data_year['bleu'].idxmax()]  # todo really names 'bleu'?
                            res_score[src_lang, trg_lang].append(
                                data_year.loc[data_year['bleu'].idxmax()][col_name])
                        # if best_bleu:
                        #     print('-->', type(data_year[col_name]))
                        #     res_score[src_lang, trg_lang].append(
                        #         data_year.loc[data_year[col_name].idxmax()][col_name])
                        else:
                            res_score[src_lang, trg_lang].append(
                                data_year.loc[data_year[col_name].idxmin()][col_name])
        for s, t in res_year.keys():
            curr_years = res_year[s, t]
            if 'wmt20' in curr_years and curr_years[-1] != 'wmt20':
                print('CURR YEARS->', s, t, curr_years)
                raise RuntimeError('years unsorted '+str(years))
            if len(res_year[s, t]) == 1:
                plt.scatter(res_year[s, t], np.array(res_score[s, t]), label=s)
            elif len(res_year[s, t]) > 1:
                plt.plot(res_year[s, t], np.array(res_score[s, t]), label=s)
        plt.xlabel('year')
        plt.ylabel('score')
        plt.suptitle(score_name)
        plt.title(name)
        # plt.legend(loc=0)
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
        plt.tight_layout()
        by = 'best_bleu' if best_bleu else 'best_score'
        plt.savefig(os.path.join(save_dir, name + '_' + by + '.png'),
                    bbox_inches='tight')
        plt.clf()
        plt.close()


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
                    data_year = data_year.loc[
                        data_year['bleu'].idxmax()]  # todo really names 'bleu'?
                for name, col_name in zip(names, score_columns):
                    if np.min(data_year[col_name]) == float('inf'):
                        continue
                    res_score[name].append(np.min(data_year[col_name]))
                    res_year[name].append(year)
                    continue
                    if type(col_name) is list:  # the score is a sum of columns
                        total = 0
                        for c in col_name:
                            total += data_year.loc[data_year[c].idxmin()][c]
                        res_score[name].append(total)
                        raise RuntimeError('why still list?')
                    else:
                        # res_score[name].append(
                        #     data_year.loc[data_year[col_name].idxmin()][col_name])
                        if np.min(data_year[col_name]) == float('inf'):
                            continue
                        res_score[name].append(np.min(data_year[col_name]))
            plot = False
            for name in res_year:
                plot = True
                if len(res_year[name]) == 1:
                    plt.scatter(res_year[name], np.array(res_score[name]), label=name)
                elif len(res_year[name]) > 1:
                    plt.plot(res_year[name], np.array(res_score[name]), label=name)
                else:
                    raise RuntimeError('no data')
            if not plot:
                continue
            plt.xlabel('year')
            plt.ylabel('score')
            plt.suptitle(score_name)
            plt.title(src_lang + trg_lang)
            # plt.legend(loc=0)
            plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
            plt.tight_layout()
            by = 'best_bleu' if best_bleu else 'best_score'
            plt.savefig(os.path.join(save_dir, src_lang + trg_lang + '_' + by + '.png'),
                        bbox_inches='tight')
            plt.clf()
            plt.close()


def score_vs_bleu(data_path, first_score_name, second_score_name, src=None, trg='en',
                  save_dir='analysis/score_vs_bleu',
                  best_bleu=False):
    save_dir = os.path.join(save_dir, score2dir(first_score_name))
    os.makedirs(save_dir, exist_ok=True)
    all_data = get_data(data_path)
    for src_lang in get_lang(src, all_data, 'src'):
        for trg_lang in get_lang(trg, all_data, 'trg'):
            data = all_data[all_data['src'] == src_lang]
            data = data[data['trg'] == trg_lang]

            years = np.unique(data['year'])
            res_score = defaultdict(list)
            res_year = defaultdict(list)
            first_name, first_columns = get_column_names(first_score_name, data.columns)
            second_name, second_columns = get_column_names(second_score_name, data.columns)
            data_year = data[data[first_columns] < float('inf')]
            if data_year.empty:
                continue
            # for year in years:
            #     data_year = data[data['year'] == year]
            #     if data_year.empty:
            #         continue

            plt.scatter(data_year[first_columns], data_year[second_columns])
            plt.xlabel(first_name)
            plt.ylabel(second_name)
            # plt.suptitle(year)
            plt.title(src_lang + trg_lang)
            # plt.legend(loc=0)
            plt.savefig(os.path.join(save_dir, src_lang + trg_lang
                                     + '_' + first_score_name + '_' + second_score_name + '.png'),
                        bbox_inches='tight')
            plt.clf()
            plt.close()

def corr_entry(src_lang, trg_lang, year_name, all_names, all_columns, data):
    entry = {'src': src_lang, 'trg': trg_lang, 'year': year_name}
    for i in range(len(all_names)):
        for j in range(i + 1, len(all_names)):
            a = np.array(data[all_columns[i]])
            b = np.array(data[all_columns[j]])
            places = (~np.isnan(a)) & (a != float('inf')) & (~np.isnan(b)) & (b != float(
                'inf'))
            a = a[places]
            b = b[places]
            if len(a) <= 1:
                continue
            r = pearsonr(a, b)[0]
            entry['num_data'] = len(a)
            entry[all_names[i] + '_vs_' + all_names[j]] = r
    return entry

def correlation(data_path, src=None, trg='en',
                 save_dir='analysis'):
    save_dir = os.path.join(save_dir, 'correlation')
    os.makedirs(save_dir, exist_ok=True)
    all_data = get_data(data_path)
    all_names = ['bleu']
    all_columns = ['bleu']
    for score_name in ['pos', 'feat', 'ner','scorer']:
        names, score_columns = get_column_names(score_name, all_data.columns)
        all_names += names
        all_columns += score_columns


    res = pd.DataFrame(columns=['src', 'trg', 'year', 'num_data'])
    for src_lang in get_lang(src, all_data, 'src'):
        for trg_lang in get_lang(trg, all_data, 'trg'):
            data = all_data[all_data['src'] == src_lang]
            data = data[data['trg'] == trg_lang]
            years = np.unique(data['year'])



            entry = corr_entry(src_lang, trg_lang, 'all', all_names,
                               all_columns, data)
            res = res.append(entry, ignore_index=True)

            for year in years:
                entry = corr_entry(src_lang, trg_lang, year, all_names,
                                   all_columns, data[data['year']==year])
                res = res.append(entry, ignore_index=True)

    res.to_csv(os.path.join(save_dir, 'correlations_r.csv'), index=False)






if __name__ == '__main__':
    path = 'scores_metrics_best_bleu.csv'
    correlation(data_path=path)
    # import sys
    # sys.exit()
    for bb in [True]:
        for t in ['pos', 'feat', 'scorer']:
            # score_by_year(data_path=path, score_name='masking')
            score_by_year(data_path=path, score_name=t, best_bleu=bb)
            # score_by_year(data_path=path, score_name='ner', best_bleu=bb)
            # score_by_year(data_path=path, score_name='feat', best_bleu=bb)
            # score_by_year(data_path=path, score_name='scorer', best_bleu=bb)
            lang_by_year(data_path=path, score_name=t, best_bleu=bb) #src=['de', 'cs', 'ru',
            # 'tr', 'zh','fi'],
            # lang_by_year(data_path=path, score_name='masking', src=['de', 'cs',
            #                                                                                'ru', 'tr', 'zh','fi'])
            # lang_by_year(data_path=path, score_name='ner', src=['de', 'cs', 'ru',
            #                                                                            'tr', 'zh','fi'])
            # lang_by_year(data_path=path, score_name='pos', src=['de', 'cs', 'ru',
            #                                                                            'tr', 'zh','fi'], best_bleu=bb)
            # lang_by_year(data_path=path, score_name='feat', src=['de', 'cs', 'ru',
            #                                                                            'tr', 'zh','fi'], best_bleu=bb)
    # score_vs_bleu(data_path=path, first_score_name='sentiment',
    #                       second_score_name='concreteness')
    # score_vs_bleu(data_path=path, first_score_name='bleu',
    #                       second_score_name='sentiment')
    # score_vs_bleu(data_path=path, first_score_name='bleu',
    #                       second_score_name='sentiment')
