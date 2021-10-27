import os
import math
import librosa
import warnings
import pandas as pd
import numpy as np

from datasets import Dataset


def speech_file_to_array(x):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        speech_array, sampling_rate = librosa.load(x, sr=16_000)
    return speech_array


def import_hu(location, test_train):
    print('Import HU dataset')
    
    if not test_train:
        dfs = []

        speakers = os.listdir(location + '/labels')
        
        for csv_file in speakers:
            speaker_id = csv_file[:-4]

            df = pd.read_csv(location + '/labels/' + csv_file)
            
            df['speech'] = [speech_file_to_array(os.path.join(location, x)) for x in df['path']]
            df['id'] = [speaker_id for x in df['path']]

            df.rename(columns={'transcription': 'target'}, inplace=True)

            df.drop('duration_in_s', axis=1, inplace=True)
            df.drop('speech_type', axis=1, inplace=True)
            df.drop('severity', axis=1, inplace=True)
            df.drop('quality', axis=1, inplace=True)
            df.drop('audioID', axis=1, inplace=True)
            df.drop('path', axis=1, inplace=True)

            dfs.append(Dataset.from_pandas(df))
        
        return dfs

    if test_train:
        train_ds = pd.DataFrame(columns=['id', 'speech', 'target'])
        test_ds = pd.DataFrame(columns=['id', 'speech', 'target'])

        speakers = os.listdir(location + '/labels')

        for csv_file in speakers:
            speaker_id = csv_file[:-4]

            df = pd.read_csv(location + '/labels/' + csv_file)

            df['speech'] = [speech_file_to_array(
                os.path.join(location, x)) for x in df['path']]
            df['id'] = [speaker_id for x in df['path']]

            df['target'] = [str(x) for x in df['transcription']]

            df.drop('transcription', axis=1, inplace=True)
            df.drop('duration_in_s', axis=1, inplace=True)
            df.drop('speech_type', axis=1, inplace=True)
            df.drop('severity', axis=1, inplace=True)
            df.drop('quality', axis=1, inplace=True)
            df.drop('audioID', axis=1, inplace=True)
            df.drop('path', axis=1, inplace=True)

            tr_ds, te_ds = np.split(df, [math.ceil(int(.8*len(df)))])

            train_ds = train_ds.append(tr_ds)
            test_ds = test_ds.append(te_ds)

        return Dataset.from_pandas(train_ds), Dataset.from_pandas(test_ds)

