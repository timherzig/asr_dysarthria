import os
import librosa
import warnings
import pandas as pd

from datasets import Dataset


def speech_file_to_array(x):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        speech_array, sampling_rate = librosa.load(x, sr=16_000)
    return speech_array


def import_hu(location):
    print('Import HU dataset')
    dfs = []

    speakers = os.listdir(location + '/labels')
    
    cnt = 0
    for csv_file in speakers:
        if cnt > 1: return dfs
        cnt += 1
        speaker_id = csv_file[6:-4]

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
