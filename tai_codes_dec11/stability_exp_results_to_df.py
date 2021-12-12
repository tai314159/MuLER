import pickle
import numpy as np
import pandas as pd
import os

def load_pickle(filepath):
    with open(filepath, 'rb') as file:
        loaded_file = pickle.load(file)
    return loaded_file

def save_pickle(out_filepath,file_to_save):
    with open(out_filepath, 'wb') as fp:
        pickle.dump(file_to_save, fp)

#######################################

DIR_RESULTS = '/cs/snapless/oabend/tailin/MT/NEW/stability_results/'

splits = ["0.3","0.4","0.5"] #different df for each split

trg_lang = "en"
years = [15, 18,19, 20,20]
src_langs = ["fi", "ru", "de","de","ru"]



main_cols = ['year','langs','submission','system_bleu']
other_cols = ['new_muler','max_bleu','min_bleu','hyb_bleu','bleu_indices']
POS = ['NOUN','VERB']


for split in splits:

    DIR_RESULTS_FINAL = DIR_RESULTS + split + "/"

    all_rows = []

    # build header
    header = []
    h = []
    for pos in POS:
        for col_name in other_cols:
            h.append(pos + "_" + col_name)
    header = main_cols + h
    print("header", header)

    for filename in os.listdir(DIR_RESULTS_FINAL):

        print("filename",filename)

        if filename.endswith(".txt"):

            d = load_pickle(DIR_RESULTS_FINAL + filename)
            row_results = []

            row_results.append(d['year'])
            row_results.append(d['langs'])
            row_results.append(d['submission'])
            row_results.append(d['results'][0]['bleu_not_masked'])  # system bleu


            for pos in POS:
                row_results.append(d['results'][0]['bleu'][pos]) #new muler
                row_results.append(d['results'][0]['max_bleu'][pos]) #max bleu
                row_results.append(d['results'][0]['min_bleu'][pos]) #min bleu
                row_results.append(d['results'][0]['hyb_bleu'][pos]) #hyb bleu
                row_results.append(d['results'][0]['bleu_indices'][pos])

            all_rows.append(row_results)

    #create the dataframe from rows
    final_df = pd.DataFrame(all_rows, columns=header)

    #save dataframe as csv
    final_df.to_csv("/cs/snapless/oabend/tailin/MT/NEW/stability_results/"+"stability_exp_"+split+".csv")

