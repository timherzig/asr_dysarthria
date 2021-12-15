import os
import math
import librosa
import warnings
import pandas as pd
import numpy as np

from datasets import Dataset
from pandas.core.frame import DataFrame

def speech_file_to_array(x):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        speech_array, sampling_rate = librosa.load(x, sr=16_000)
    return speech_array

def duration(x):
    return librosa.get_duration(filename=x)

def import_hu_split(location, test_train, train_split, test_split):
    print('Import HU dataset')

    df = pd.read_csv(location + '/labels.csv')
    df['speech'] = [speech_file_to_array(
        os.path.join(location, 'split' + str(y), x)) for x, y in zip(df['path'], df['split'])]
    df['len'] = [duration(
        os.path.join(location, 'split' + str(y), x)) for x, y in zip(df['path'], df['split'])]

    df.rename(columns={'spkID': 'id'}, inplace=True)
    df.rename(columns={'transcription': 'target'}, inplace=True)

    df.drop('duration_in_s', axis=1, inplace=True)
    df.drop('speech_type', axis=1, inplace=True)
    df.drop('severity', axis=1, inplace=True)
    df.drop('quality', axis=1, inplace=True)
    df.drop('audioID', axis=1, inplace=True)
    df.drop('gender', axis=1, inplace=True)
    df.drop('path', axis=1, inplace=True)
    df.drop('age', axis=1, inplace=True)

    if test_train == 'train':
        print('TRAIN')
        
        tr_ds0 = df[df['split'] == int(train_split[1])]
        tr_ds1 = df[df['split'] == int(train_split[0])]
        tr_ds = pd.concat([tr_ds0, tr_ds1])
        
        te_ds = df[df['split'] == int(test_split)]
        
        tr_ds.drop('split', axis=1, inplace=True)
        te_ds.drop('split', axis=1, inplace=True)

        return Dataset.from_pandas(tr_ds), Dataset.from_pandas(te_ds)
        
    elif test_train == 'test':
        print('TEST')

        te_ds = df[df['split'] == int(test_split)]
        te_ds.drop('split', axis=1, inplace=True)

        dfs = []

        for speaker in te_ds['id'].unique():
            s_ds = te_ds[te_ds['id'] == speaker]
            dfs.append(Dataset.from_pandas(s_ds))
        
        for df in dfs:
            print(df['id'][0])

        return dfs
