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
    speakers = [x[0] for x in os.walk(location)]
    df = pd.DataFrame(columns=['id', 'file', 'target'])

    speakers.pop(0)
    for speaker in speakers:
        files = os.listdir(speaker)
        file_path = ''
        transcription = ''

        for file in files:
            if file[-17:] == 'transcription.txt':
                transcription = open(
                    os.path.join(speaker, file), 'r').read()
            if file[-4:] == '.mp3':
                file_path = os.path.join(speaker, file)

        new_row = {'id': speaker[6:], 'file': file_path,
                    'target': transcription}
        df = df.append(new_row, ignore_index=True)

    df['speech'] = [speech_file_to_array(x) for x in df['file']]
    df.drop('file', axis=1, inplace=True)

    print(df.columns)

    return dfs.append(Dataset.from_pandas(df))
