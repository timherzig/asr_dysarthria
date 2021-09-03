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


def split_handle(handle, audio_location):
    speaker_id, book_id, audio_id = handle.split('_')
    return os.path.join(audio_location, speaker_id, book_id, handle + '.opus')


def import_mls_de(handle_location, transcription_location, audio_location):
    print('Import MLS dataset')
    dfs = []

    h = pd.read_csv(handle_location, delimiter="\t", header=None)
    if(len(h.columns) >= 2):
        h.drop(h.columns[[1, 2, 3]], axis=1, inplace=True)
    ts = pd.read_csv(transcription_location, delimiter="\t", header=None)

    ts.columns = ['id', 'sentence']
    h.columns = ['id']

    df = pd.merge(ts, h, left_on='id', right_on='id')
    df['file'] = [split_handle(x, audio_location) for x in df['id']]
    df['speech'] = [speech_file_to_array(x) for x in df['file']]
    df.drop('file', axis=1, inplace=True)
    df.rename(columns={'sentence': 'target'}, inplace=True)

    print(df.columns)

    return dfs.append(Dataset.from_pandas(df))
