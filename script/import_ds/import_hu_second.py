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
    final_speakers = ['F05', 'M01', 'M02']
    dfs = []

    speakers = os.listdir(location)
    
    for speaker in speakers:
        if speaker in final_speakers:
            df_list = []
            cur_location = os.path.join(location, speaker)
            in_cur_l = os.listdir(cur_location)
            for icl in in_cur_l:
                #tmp_df = pd.DataFrame(columns=['start', 'end', 'duration', 'transcription', 'speech_type', 'quality', 'severity', 'raw_audio', 'id', 'file'])
                if icl.endswith('.csv'):
                    if icl.endswith('therapist.csv'):
                        continue
                    if icl.endswith('patient.csv'):
                        tmp_df = pd.read_csv(os.path.join(cur_location, icl))
                        tmp_df['speech'] = [speech_file_to_array(cur_location + '/' + icl[:-12] + '/' +  x) for x in tmp_df['file']]
                        df_list.append(tmp_df)
                    else:
                        tmp_df = pd.read_csv(os.path.join(cur_location, icl))
                        tmp_df['speech'] = [speech_file_to_array(cur_location + '/' + icl[:-4] + '/copy_' +  x) for x in tmp_df['file']]
                        df_list.append(tmp_df)
            
            df = pd.concat(df_list)
            df.drop('start', axis=1, inplace=True)
            df.drop('end', axis=1, inplace=True)
            df.drop('duration', axis=1, inplace=True)
            df.drop('severity', axis=1, inplace=True)
            df.drop('speech_type', axis=1, inplace=True)
            df.drop('quality', axis=1, inplace=True)
            if 'raw_audio' in df.columns: df.drop('raw_audio', axis=1, inplace=True) 
            else: df.drop('audio', axis=1, inplace=True)
            df.drop('file', axis=1, inplace=True)
            df.rename(columns={'transcription': 'target'}, inplace=True)

            dfs.append(Dataset.from_pandas(df))
    
    return dfs
