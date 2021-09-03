import re
import soundfile as sf
from datasets import load_dataset


def map_to_array(batch):
    speech, _ = sf.read(batch['file'], samplerate=16000)
    batch['speech'] = speech
    return batch

def import_mls_en(percentage):
    tds = []
    df = load_dataset('librispeech_asr', 'clean', split='test[:{}%]'.format(percentage))
    df = df.map(map_to_array)

    print(df.columns)

    return tds.append(df)
