import pandas as pd
import numpy as np
import os
import warnings
from collections import defaultdict
from evaluate import eval

def path2list(filepath):
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
    if len(txts) == 1:
        return txts[0]
    return txts



def get_names(possibilities, required, name_mapping=lambda x: x):
    if required is None:
        return possibilities
    names = []
    for r in required:
        name = name_mapping(r)
        if name in possibilities:
            names.append(name)
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

    possible_years = filter(lambda x: x.startswith('wmt'), os.listdir(submissions_path))
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

        possible_src = map(lambda x: x[:2], os.listdir(candidates_dir))
        possible_trg = map(lambda x: x[-2:], os.listdir(candidates_dir))

        src_names = get_names(possible_src, sources)
        trg_names = get_names(possible_trg, targets)
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
                collection[year][src, trg] = reference_path, candidate_paths
    return collection


def create_database(submissions_path, output_path, metrics=None, years=None, sources=None,
                    targets=None):
    paths = iterate_submissions(submissions_path, years, sources, targets)
    columns = ['year', 'src', 'trg', 'submission', 'sentence_bleu']
    # for metric in metrics:
    #     columns.append(metric)
    database = pd.DataFrame(columns=columns)
    for year in paths:
        for src, trg in paths[year]:
            print('-->', year, src, trg)
            ref_path, can_paths = paths[year][src, trg]

            ref, cans = path2list(ref_path), path2list(can_paths)
            results = eval(ref, cans, trg)

            for i, can in enumerate(cans):
                submission_name = can_paths[i].split('/')[-1]
                entry = {'year': year, 'src': src, 'trg': trg, 'submission': submission_name}
                entry.update(results[i])
                database = database.append(entry, ignore_index=True)
            break
    database.to_csv(os.path.join(output_path, 'scores.csv'), index=False)


if __name__ == '__main__':
    submissions_path = '/cs/snapless/oabend/borgr/SSMT/data/submissions'
    collection = iterate_submissions(submissions_path,
                                     targets='en')
    print(len(collection['wmt14']['de', 'en']))
    ref, cans = collection['wmt14']['de', 'en']
    print(ref.split('/')[-1])
    print(cans[0].split('/')[-1])
    print('-')
    print(collection.keys())
    print(collection['wmt14']['fr', 'en'][0])
    print(collection['wmt14']['fr', 'en'][1][0])

    create_database(submissions_path, '/cs/labs/oabend/gal.patel/projects/MT_eval', years=19,
                    targets='en')