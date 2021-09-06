import re
import soundfile as sf
from datasets import load_dataset


def map_to_array(batch):
    speech, _ = sf.read(batch['path'], samplerate=16000)
    batch['speech'] = speech
    return batch

def import_cv(language, percentage):
    tds = []
    df = load_dataset('common_voice', language, 'clean', split='test[:{}%]'.format(percentage))
    df = df.map(map_to_array)

    df.drop('path', axis=1, inplace=True)
    df.drop('client_id', axis=1, inplace=True)
    df.drop('up_votes', axis=1, inplace=True)
    df.drop('down_votes', axis=1, inplace=True)
    df.drop('age', axis=1, inplace=True)
    df.drop('gender', axis=1, inplace=True)
    df.drop('accent', axis=1, inplace=True)
    df.drop('locale', axis=1, inplace=True)
    df.drop('segment', axis=1, inplace=True)

    df.rename(columns={'sentence': 'target'}, inplace=True)


    print(df.columns)

    return tds.append(df)
