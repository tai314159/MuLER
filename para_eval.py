# analyzing paraphrases
from evaluate import eval
import pandas as pd

class DataReader:
    def __init__(self,
                 csv_path='/cs/labs/oabend/gal.patel/analyzing_neurons/mydata_filtered_add_trans.csv',
                 clause=True):

        data = pd.read_csv(csv_path)
        data = data[data['clause'] == clause]
        self.orig = list(data['sentence'])
        self.ref = list(data['translation'])
        if clause:
            self.para = list(data['para_cl'])
            self.orig_type = 'Clause'
            self.para_type = 'Noun Phrase'
        else:
            self.para = list(data['para_pas'])
            self.orig_type = 'Active'
            self.para_type = 'Passive'


    def __len__(self):
        return len(self.orig)

if __name__ == '__main__':
    database = pd.DataFrame()
    for data in ['clause_nphrase', 'active_passive']:
        reader = DataReader(clause=data.startswith('clause'))
        res = eval(reader.orig, [reader.para])
        entry = {'data': data}
        entry.update(res)
        database = database.append(entry, ignore_index=True)
    database.to_csv('paraphrase_metrics.csv', index=False)