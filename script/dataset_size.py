import librosa

from datasets import Dataset
from import_ds.import_dataset import import_dataset

severities = {'vl': ['F03', 'M14', 'M17'], 'l': ['F02', 'F05', 'F06', 'M01', 'M09', 'M12', 'M20'], 'm': ['M01', 'M03', 'M06', 'M10', 'M15'], 'h': ['F01', 'F04', 'M02', 'M04', 'M08', 'M16']}
times = {'vl': [0, 0], 'l': [0, 0], 'm': [0, 0], 'h': [0, 0]}


train, test = import_dataset('hu 2 3 1', True, True)


for row in train:
    if row['id'] in severities['vl']:
        times['vl'][0] = times['vl'][0] + row['len']

    elif row['id'] in severities['l']:
        times['l'][0] = times['l'][0] + row['len']

    elif row['id'] in severities['m']:
        times['m'][0] = times['m'][0] + row['len']

    elif row['id'] in severities['h']:
        times['h'][0] = times['h'][0] + row['len']


for row in test:
    if row['id'] in severities['vl']:
        times['vl'][1] = times['vl'][1] + row['len']

    elif row['id'] in severities['l']:
        times['l'][1] = times['l'][1] + row['len']
    
    elif row['id'] in severities['m']:
        times['m'][1] = times['m'][1] + row['len']
    
    elif row['id'] in severities['h']:
        times['h'][1] = times['h'][1] + row['len']


for t in times:
    print(t + ' train: ' + str(times[t][0]))
    print(t + ' test: ' + str(times[t][1]))
    