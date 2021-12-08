from __future__ import absolute_import, division, print_function

import argparse
import numpy as np
import shlex
import subprocess
import sys
import wave
import json

from deepspeech import Model, version
from timeit import default_timer as timer

try:
    from shhlex import quote
except ImportError:
    from pipes import quote



def convert_samplerate(audio_path, desired_sample_rate):
    sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate {} --encoding signed-integer --endian little --compression 0.0 --no-dither - '.format(
        quote(audio_path), desired_sample_rate)
    try:
        output = subprocess.check_output(
            shlex.split(sox_cmd), stderr=subprocess.PIPE)
    except subprocess.CalledProcessError as e:
        raise RuntimeError('SoX returned non-zero status: {}'.format(e.stderr))
    except OSError as e:
        raise OSError(e.errno, 'SoX not found, use {}hz files or install it: {}'.format(
            desired_sample_rate, e.strerror))

    return desired_sample_rate, np.frombuffer(output, np.int16)

def stt(model, scorer, audio):
    m = Model(model)
    sr = m.sampleRate()
    m.enableExternalScorer(scorer)

    fin = wave.open(audio, 'rb')
    fs_orig = fin.getframerate()
    if fs_orig != sr:
        print('Warning: original sample rate ({}) is different than {}hz. Resampling might produce erratic speech recognition.'.format(
            fs_orig, sr), file=sys.stderr)
        fs_new, a = convert_samplerate(audio, sr)
    else:
        a = np.frombuffer(fin.readframes(fin.getnframes()), np.int16)

    print(m.stt(a))


stt('/home/tim/Documents/Models/deepspeech/deepspeech-0.9.3-models.pbmm', '/home/tim/Documents/Models/deepspeech/deepspeech-0.9.3-models.scorer',
    '/home/tim/Documents/Datasets/torgo/TORGO/MC04/Session1/wav_arrayMic/0010.wav')
