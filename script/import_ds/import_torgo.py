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


def import_torgo(location):
    print('Import Torgo dataset')
    speakers = os.listdir(location)
    dfs = []

    for speaker in speakers:
        df = pd.DataFrame(columns=['id', 'file', 'target'])
        l = os.path.join(location, speaker)
        if os.path.isdir(l):
            sessions = os.listdir(l)
            for session in sessions:
                if session[0:7] == 'Session':
                    s = os.path.join(l, session)

                    if os.path.isdir(os.path.join(s, 'wav_arrayMic')):
                        recordings_location = os.path.join(
                            s, 'wav_arrayMic')
                    else:
                        recordings_location = os.path.join(
                            s, 'wav_headMic')

                    recordings = os.listdir(recordings_location)
                    for recording in recordings:
                        if len(recording) == 8:
                            if os.path.isfile(os.path.join(s, 'prompts', (recording[:-4] + '.txt'))):
                                sentence = open(os.path.join(
                                    s, 'prompts', (recording[:-4] + '.txt'))).read()
                                if "[" in sentence or "/" in sentence or "xxx" in sentence:
                                    continue
                                else:
                                    new_row = {'id': speaker, 'file': os.path.join(
                                        recordings_location, recording), 'target': sentence}
                                    df = df.append(
                                        new_row, ignore_index=True)
                            else:
                                continue

            df['speech'] = [speech_file_to_array(x) for x in df['file']]
            
            df.drop('file', axis=1, inplace=True)

            dfs.append(Dataset.from_pandas(df))

    return dfs
