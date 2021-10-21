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


def import_ua(location, test_train):
    print('Import UASpeech dataset')
    
    if not test_train:
        dfs = []

        xls = pd.ExcelFile(os.path.join(location, 'speaker_wordlist.xls'))
        words = pd.read_excel(xls, 'Word_filename')

        # FOR PATIENTS
        speakers = os.listdir(os.path.join(location, 'audio'))
        speakers.remove('control')
        for speaker in speakers:
            df = pd.DataFrame(columns=['id', 'speech', 'target'])

            files = os.listdir(os.path.join(location, 'audio', speaker))
            for file in files:
                id, block, word_id = [file.split('_')[i] for i in (0, 1, 2)]
                word = words.loc[words['FILE NAME'] == word_id]
                if word.empty:
                    word = words.loc[words['FILE NAME'] == (block + '_' + word_id)]
                target = str(word.iloc[0]['WORD'])
                speech = speech_file_to_array(os.path.join(location, 'audio', speaker, file))

                df = df.append({'id': id, 'target': target,'speech': speech}, ignore_index=True)

            dfs.append(Dataset.from_pandas(df))

        # FOR CONTROL SPEAKERS 
        cspeakers = os.listdir(os.path.join(location, 'audio', 'control'))
        for speaker in cspeakers:
            cdf = pd.DataFrame(columns=['id', 'speech', 'target'])

            files = os.listdir(os.path.join(location, 'audio', 'control', speaker))
            for file in files:
                id, block, word_id = [file.split('_')[i] for i in (0, 1, 2)]
                word = words.loc[words['FILE NAME'] == word_id]
                if word.empty:
                    word = words.loc[words['FILE NAME'] == (block + '_' + word_id)]
                target = str(word.iloc[0]['WORD'])
                speech = speech_file_to_array(
                    os.path.join(location, 'audio', 'control', speaker, file))

                cdf = cdf.append({'id': id, 'target': target,
                            'speech': speech}, ignore_index=True)
            
            dfs.append(Dataset.from_pandas(cdf))

        return dfs

    if test_train:
        train_ds = pd.DataFrame(columns=['id', 'speech', 'target'])
        test_ds = pd.DataFrame(columns=['id', 'speech', 'target'])

        xls = pd.ExcelFile(os.path.join(location, 'speaker_wordlist.xls'))
        words = pd.read_excel(xls, 'Word_filename')

        # FOR PATIENTS
        speakers = os.listdir(os.path.join(location, 'audio'))
        speakers.remove('control')
        for speaker in speakers:
            files = os.listdir(os.path.join(location, 'audio', speaker))
            for file in files:
                id, block, word_id = [file.split('_')[i] for i in (0, 1, 2)]
                word = words.loc[words['FILE NAME'] == word_id]
                if word.empty:
                    word = words.loc[words['FILE NAME'] == (block + '_' + word_id)]
                target = str(word.iloc[0]['WORD'])
                speech = speech_file_to_array(
                    os.path.join(location, 'audio', speaker, file))

                if block == ('B1' or 'B3'):
                    train_ds = train_ds.append(
                        {'id': id, 'target': target, 'speech': speech}, ignore_index=True)
                elif block == 'B2':
                    test_ds = test_ds.append(
                        {'id': id, 'target': target, 'speech': speech}, ignore_index=True)

        # FOR CONTROL SPEAKERS
        cspeakers = os.listdir(os.path.join(location, 'audio', 'control'))
        for speaker in cspeakers:
            cdf = pd.DataFrame(columns=['id', 'speech', 'target'])

            files = os.listdir(os.path.join(
                location, 'audio', 'control', speaker))
            for file in files:
                id, block, word_id = [file.split('_')[i] for i in (0, 1, 2)]
                word = words.loc[words['FILE NAME'] == word_id]
                if word.empty:
                    word = words.loc[words['FILE NAME'] == (block + '_' + word_id)]
                target = str(word.iloc[0]['WORD'])
                speech = speech_file_to_array(
                    os.path.join(location, 'audio', 'control', speaker, file))

                if block == ('B1' or 'B3'):
                    train_ds = train_ds.append(
                        {'id': id, 'target': target, 'speech': speech}, ignore_index=True)
                elif block == 'B2':
                    test_ds = test_ds.append(
                        {'id': id, 'target': target, 'speech': speech}, ignore_index=True)
        
        return Dataset.from_pandas(train_ds), Dataset.from_pandas(test_ds)
