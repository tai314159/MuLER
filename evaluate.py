import pandas
import numpy as np
import os
import warnings


def path2list(filepath):
    """
    Convert from a file path in standart form (each line is a text item) to a list
    """
    with open(filepath, 'r') as txtfile:
        for line in txtfile:
            line = line.rstrip("\n")
            txt_list.append(line)
    return txt_list


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


    collection = dict()

    for year in year_names:
        collection[year] = dict()
        full_year = '20' + year[-2:]
        candidates_dir = os.path.join(submissions_path, year, 'plain', 'system-outputs')
        references_dir = os.path.join(submissions_path, year, 'plain', 'references')

        # some has another direstory
        if 'newstest' + full_year in os.listdir(candidates_dir):
            candidates_dir = os.path.join(candidates_dir, 'newstest' + full_year)

        # lang_pairs = map(lambda x: x[len(full_year) + 1:len(full_year) + 5], os.listdir(
        #     candidates_dir))
        possible_src = map(lambda x: x[:2], os.listdir(
            candidates_dir))
        possible_trg = map(lambda x: x[-2:], os.listdir(
            candidates_dir))
        print(possible_src, possible_trg)

        src_names = get_names(possible_src, sources)
        trg_names = get_names(possible_trg, targets)
        references_files = os.listdir(references_dir)

        for src in src_names:
            for trg in trg_names:
                reference_path = None
                for f in references_files:
                    if src+trg in f: # todo: previous than 2014
                        reference_path = os.path.join(references_dir, f)
                        break
                lang_pair = os.path.join(candidates_dir, src+'-'+trg)
                if not os.path.isdir(lang_pair):
                    continue
                candidate_paths = list(map(lambda x: os.path.join(lang_pair, x),os.listdir(
                    lang_pair)))
                collection[year][src, trg] = reference_path, candidate_paths
    return collection

if __name__ == '__main__':
    collection = iterate_submissions('/cs/snapless/oabend/borgr/SSMT/data/submissions',
                                     targets='en')
    print(len(collection['wmt14']['de', 'en']))
    ref, cans = collection['wmt14']['de', 'en']
    print(ref.split('/')[-1])
    print(cans[0].split('/')[-1])
    print('-')