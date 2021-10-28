import math
import os
import librosa
import warnings
import numpy as np
import pandas as pd

from datasets import Dataset
from transformers.file_utils import filename_to_url

longest_audio = 0

def speech_file_to_array(x):
    global longest_audio
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        speech_array, sampling_rate = librosa.load(x, sr=16_000)
    len = librosa.get_duration(filename=x)
    longest_audio = len if len > longest_audio else longest_audio
    return speech_array


def import_torgo(location, test_train):
    print('Import Torgo dataset')

    if not test_train:
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

        print('Longest audio ' + str(longest_audio))        
        return dfs

    if test_train:
        train_ds = pd.DataFrame(columns=['id', 'speech', 'target'])
        test_ds = pd.DataFrame(columns=['id', 'speech', 'target'])

        speakers = os.listdir(location)

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

                tr_ds, te_ds = np.split(df, [math.ceil(int(.8*len(df)))])

                train_ds = train_ds.append(tr_ds)
                test_ds = test_ds.append(te_ds)
        
        print('Longest audio ' + str(longest_audio))
        return Dataset.from_pandas(train_ds), Dataset.from_pandas(test_ds)
