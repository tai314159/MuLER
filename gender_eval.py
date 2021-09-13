# analyzing gender
from evaluate import eval
import pandas as pd


class DataReader:
    def __init__(self, csv_path='winogender.tsv'):

        data = pd.read_csv(csv_path, sep='\t')
        self.female = []
        self.male = []
        self.neutral = []
        for i, row in data.iterrows():
            sentence = row['sentence']
            info = row['sentid']
            gender = info.split('.')[-2]
            if gender == 'female':
                self.female.append(sentence)
            elif gender == 'male':
                self.male.append(sentence)
            elif gender == 'neutral':
                self.neutral.append(sentence)
            else:
                raise RuntimeError('No such gender ' + gender)

    def __len__(self):
        assert len(self.female) == len(self.male)
        assert len(self.female) == len(self.neutral)
        return len(self.female)


if __name__ == '__main__':
    database = pd.DataFrame()
    reader = DataReader()
    # for pair in [('female', 'male'), ('female', 'neutral'), ('male', 'neutral')]:
    #     a, b = pair
    types = ['female', 'male', 'neutral']
    for a in types:
        for b in types:
            res = eval(getattr(reader, a), [getattr(reader, b)])
            entry = {'data': a + '_' + b}
            entry.update(res)
            database = database.append(entry, ignore_index=True)
    database.to_csv('gender_metrics.csv', index=False)
