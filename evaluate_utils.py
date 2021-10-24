import pandas as pd
import numpy as np
import os
import warnings
from collections import defaultdict
from evaluate import eval_muler
import sys

DISCARD = {'newstest2020.de-en.yolo.1052.txt'}


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





def get_names(possibilities, required, name_mapping=lambda x: x):
    # if type(possibilities) == map:
    #     possibilities = list(possibilities)
    if type(possibilities) != list:
        raise RuntimeError('possibilities must be list and got ' + str(type(possibilities)))
    if required is None:
        if len(DISCARD_SOURCES):
            return list(set(possibilities).difference(set(DISCARD_SOURCES)))
        return possibilities

    names = []
    for r in required:
        name = name_mapping(r)
        if name in possibilities:
            names.append(name)
    # if DISCARD_SOURCES:
    #     set(possibilities).difference()
    return names


def iterate_submissions(submissions_path, years=None, sources=None, targets=None):
    """
    Iterate over wmt submissions directory and returns pairs of paths for candidate and
    references files
    :param submissions_path: root directory for wmt submissions
    :param years: one or more years, integers between 0-99. If None, return every year that
    exists
    :param sources: list of source languages in str. If None, return everything
    :param targets: list of target languages in str. If None, return everything
    :return: a dictionary with keys: [year][src, trg] and value a tuple of reference path and a
    list of all submission paths
    """
    if years is not None and type(years) != list:
        years = [years]
    if sources is not None and type(sources) != list:
        sources = [sources]
    if targets is not None and type(targets) != list:
        targets = [targets]

    possible_years = list(filter(lambda x: x.startswith('wmt'), os.listdir(submissions_path)))
    year_names = get_names(possible_years, years,
                           lambda year: 'wmt0' + str(year) if year < 10 else 'wmt' + str(year))

    collection = defaultdict(dict)

    for year in year_names:
        full_year = '20' + year[-2:]
        candidates_dir = os.path.join(submissions_path, year, 'plain', 'system-outputs')
        references_dir = os.path.join(submissions_path, year, 'plain', 'references')

        # some has another direstory
        if 'newstest' + full_year in os.listdir(candidates_dir):
            candidates_dir = os.path.join(candidates_dir, 'newstest' + full_year)

        possible_src = list(map(lambda x: x[:2], os.listdir(candidates_dir)))
        possible_trg = list(map(lambda x: x[-2:], os.listdir(candidates_dir)))

        src_names = get_names(possible_src, sources)
        trg_names = get_names(possible_trg, targets)
        # print('SOURCES', year, src_names)
        references_files = os.listdir(references_dir)

        for src in src_names:
            for trg in trg_names:
                reference_path = None
                for f in references_files:
                    if src + trg in f:  # todo: previous than 2014 or (int(year[-2:]) < 14 and f.endswith(trg))
                        reference_path = os.path.join(references_dir, f)
                        break
                if reference_path is None:
                    continue
                lang_pair = os.path.join(candidates_dir, src + '-' + trg)
                if not os.path.isdir(lang_pair):
                    continue
                candidate_paths = list(map(lambda x: os.path.join(lang_pair, x), os.listdir(
                    lang_pair)))
                # candidate_paths = list(filter(lambda x: x.split('/')[-1] not in DISCARD,
                #                               candidate_paths))
                collection[year][src, trg] = reference_path, candidate_paths
    return collection


def get_exist(output_path, output_names=[]):
    output_names = ['scores_metrics_full1.csv',
                    'scores_metrics_full2.csv',
                    'scores_metrics_full3.csv',
                    'scores_metrics_full4.csv']
    exist = set()
    for output_name in output_names:
        file_name = os.path.join(output_path, output_name)
        if not os.path.isfile(file_name):
            continue
        df = pd.read_csv(file_name)
        for i, row in df.iterrows():
            exist.add((row['year'], row['src'], row['trg']))
    return exist


def merge_exist(output_path):
    output_names = ['scores_metrics_full1.csv',
                    'scores_metrics_full2.csv',
                    'scores_metrics_full3.csv',
                    'scores_metrics_full4.csv']
    missing = set()
    merged = None
    for output_name in output_names:
        file_name = os.path.join(output_path, output_name)
        if not os.path.isfile(file_name):
            continue
        df = pd.read_csv(file_name)
        if merged is None:
            merged = pd.DataFrame(columns=df.columns)
        for i, row in df.iterrows():
            details = (row['year'], row['src'], row['trg'])
            if pd.isna(row['FEAT_Definite_hallucination']):
                missing.add(details)
            else:
                missing.discard(details)
                merged = merged.append(row, ignore_index=True)
    merged.to_csv(os.path.join(output_path, 'scores_metrics_merged.csv'), index=False)
    import pickle
    with open(os.path.join(output_path, 'missing.p'), 'wb') as f:
        pickle.dump(missing, f)
    print('MISSING\n', missing)


def get_missing(output_path):
    import pickle
    with open(os.path.join(output_path, 'missing.p'), 'rb') as f:
        missing = pickle.load(f)
    return missing


def get_bestbleu(paths, knowledge_base='scores_metrics_merged.csv'):
    data = pd.read_csv(knowledge_base)
    best_paths = dict()
    for year in paths:
        best_paths[year] = dict()
        for src, trg in paths[year]:

            curr_data = data[data.year == year]
            curr_data = curr_data[curr_data.src == src]
            curr_data = curr_data[curr_data.trg == trg]
            # print('FIL', year, src, len(curr_data))
            curr_data = curr_data[~curr_data['submission'].str.contains('human')]
            curr_data = curr_data[~curr_data['submission'].str.contains('Human')]
            # print('HUMAN', year, src, len(curr_data))
            if len(curr_data) < 1:
                continue
            best_submission = curr_data.loc[curr_data['bleu'].idxmax()]['submission']
            ref_path, can_paths = paths[year][src, trg]
            new_can_paths = list(filter(lambda x: best_submission in x, can_paths))

            # print('BB:', year, src, trg, best_submission, new_can_paths)
            assert len(new_can_paths) == 1

            best_paths[year][src, trg] = ref_path, new_can_paths
    return best_paths


def get_save_name(only_missing, only_best_bleu, years):
    save_name = 'scores_metrics_base'
    if only_missing:
        save_name = 'scores_metrics_missed'
    elif only_best_bleu:
        save_name = 'scores_metrics_best_bleu'
    if years is not None:
        if type(years) == int:
            save_name += '_' + str(years)
        elif type(years) == list:
            for y in years:
                save_name += '_' + str(y)
        else:
            raise RuntimeError('No name for years of type ' + str(type(years)))
    save_name += '.csv'
    return save_name


def create_database(submissions_path, output_path, metrics=None, years=None, sources=None,
                    targets=None, only_missing=False, only_best_bleu=False):
    paths = iterate_submissions(submissions_path, years, sources, targets)
    if only_best_bleu and only_missing:
        raise RuntimeError('Can have one special treatment at a run')
    # import sys
    # sys.exit()
    columns = ['year', 'src', 'trg', 'submission', 'sentence_bleu']
    # for metric in metrics:
    #     columns.append(metric)
    database = pd.DataFrame(columns=columns)
    exist = set()  # get_exist(output_path)
    missing = None
    if only_missing:
        missing = get_missing(output_path)
    if only_best_bleu:
        print('ONLY BSET BLEU')
        paths = get_bestbleu(paths)
    save_name = get_save_name(only_missing, only_best_bleu, years)
    # import sys
    # sys.exit()
    for year in paths:
        for src, trg in paths[year]:
            if only_missing:
                if (year, src, trg) not in missing:
                    continue
            elif (year, src, trg) in exist:
                # print('skip', year, src, trg)
                continue
            print('-->', year, src, trg)
            ref_path, can_paths = paths[year][src, trg]

            # ref, cans = path2list(ref_path), path2list(can_paths, return_list_always=only_best_bleu)
            ref, cans = ref_path, can_paths

            # assert len(cans) == 1
            # assert len(cans[0]) == len(ref)
            # assert type(cans) == list
            # for i, p in enumerate(can_paths):
            #     print(i, p.split('/')[-1])
            results = eval_muler(ref, cans, trg, cache_dir=os.path.join(
                '/cs/snapless/oabend/gal.patel/MT_eval_cache/' + year + '_' + '_' + src + '_' + trg),
                                 are_paths=True)
            if len(results) != len(cans):
                print('UNMATCHED results&cans')
                database = database.append({'year': year, 'src': src, 'trg': trg},
                                           ignore_index=True)
                continue

            for i, can in enumerate(cans):
                submission_name = can_paths[i].split('/')[-1]
                entry = {'year': year, 'src': src, 'trg': trg, 'submission': submission_name}
                entry.update(results[i])
                database = database.append(entry, ignore_index=True)
            # break
            database.to_csv(os.path.join(output_path, save_name), index=False)
            # break
        print('done year', year)
        # break

    database.to_csv(os.path.join(output_path, save_name), index=False)


if __name__ == '__main__':
    submissions_path = '/cs/snapless/oabend/borgr/SSMT/data/submissions'

    # collection = iterate_submissions(submissions_path,
    #                                  targets='en')
    # print(len(collection['wmt14']['de', 'en']))
    # ref, cans = collection['wmt14']['de', 'en']
    # print(ref.split('/')[-1])
    # print(cans[0].split('/')[-1])
    # print('-')
    # print(collection.keys())
    # print(collection['wmt14']['fr', 'en'][0])
    # print(collection['wmt14']['fr', 'en'][1][0])
    if len(sys.argv) > 1:
        year = eval(sys.argv[1])
        DISCARD_SOURCES = []
    else:
        year = None
        DISCARD_SOURCES = ['de', 'cs', 'fi', 'ru', 'tr', 'zh']
    create_database(submissions_path, '/cs/labs/oabend/gal.patel/projects/MT_eval',
                    sources=None, targets='en', years=year,
                    only_missing=False, only_best_bleu=False)
    # merge_exist('/cs/labs/oabend/gal.patel/projects/MT_eval')
