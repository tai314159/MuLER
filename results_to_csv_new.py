import os
import pandas as pd
import pickle
import csv

def load_pickle(filepath):
    with open(filepath, 'rb') as file:
        loaded_file = pickle.load(file)
    return loaded_file

########################################################################################################################
#write results to csv

#ARGS
years = [2016,2017] #what years you want
mask_types = ["feat","pos","ner"] #these are currently all the mask_types we have results for
DIR_RESULTS = "/cs/snapless/oabend/tailin/MT/NEW/outputs/20.07.21/final_results/" #the results (each file is a pickled dictionary)
DIR_OUT_CSV = "/cs/snapless/oabend/tailin/MT/NEW/outputs/20.07.21/csv/" #where you want to save the csv

for mask_type in mask_types:
    ###################
    #get info for header

    dict_results_general = load_pickle(DIR_RESULTS+"newstest"+str(years[0])+ "-ruen-ref_"+mask_type+"_results_bleu.txt") #change later to randomly choose one of the files

    header_keys = []

    for key in dict_results_general["bleu"].keys():
        header_keys.append(key)

    #build header
    header = ["year", "langs", "bleu_not_masked"]

    for key1 in header_keys:
        header.append(str(key1) + "_bleu")
        header.append(str(key1) + "_hallucination:add")
        header.append(str(key1) + "_hallucination:miss")
        header.append(str(key1) + "_hallucination:hit")

    rows_final = []

    rows_final.append(header) #trial

    for year in years:

        for filename in os.listdir(DIR_RESULTS):

            if filename.startswith("newstest"+str(year)):

                if filename.endswith(mask_type +"_results_bleu.txt"):

                    rows = []

                    print("filename",filename)

                    rows.append(filename.split("-")[0])
                    rows.append(filename.split("-")[1])

                    dict_results = load_pickle(DIR_RESULTS + filename)

                    rows.append(dict_results["bleu_not_masked"])

                    for tag_key in dict_results["bleu"]:
                        rows.append(dict_results["bleu"][tag_key])
                        rows.append(dict_results["hallucination"][tag_key][0])
                        rows.append(dict_results["hallucination"][tag_key][1])
                        rows.append(dict_results["hallucination"][tag_key][2])

                    rows_final.append(rows)

    #write to csv

    with open(DIR_OUT_CSV + mask_type + "_mt_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows_final)
