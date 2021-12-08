import matplotlib.pyplot as plt
import librosa
from IPython.display import display, Audio
from espnet2.bin.asr_inference import Speech2Text
from espnet_model_zoo.downloader import ModelDownloader
import string
import torch
import time


lang = 'en'
fs = 16000  # @param {type:"integer"}
# @param ["Shinji Watanabe/spgispeech_asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_unnorm_bpe5000_valid.acc.ave", "kamo-naoyuki/librispeech_asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_bpe5000_scheduler_confwarmup_steps40000_optim_conflr0.0025_sp_valid.acc.ave"] {type:"string"}
tag = 'kamo-naoyuki/librispeech_asr_train_asr_conformer6_n_fft512_hop_length256_raw_en_bpe5000_scheduler_confwarmup_steps40000_optim_conflr0.0025_sp_valid.acc.ave'
file = '/home/tim/Documents/Datasets/hu_final/data/audio_7/audio_7_1.wav'


d = ModelDownloader()
# It may takes a while to download and build models
speech2text = Speech2Text(
    **d.download_and_unpack(tag),
    device="cpu",
    minlenratio=0.0,
    maxlenratio=0.0,
    ctc_weight=0.3,
    beam_size=10,
    batch_size=0,
    nbest=1
)


def text_normalizer(text):
    text = text.upper()
    return text.translate(str.maketrans('', '', string.punctuation))


def s2t(file_name):
  speech, rate = librosa.load(file_name, sr=16_000)
  assert rate == fs, "mismatch in sampling rate"
  nbests = speech2text(speech)
  text, *_ = nbests[0]

  print(f"Input Speech: {file_name}")
  display(Audio(speech, rate=rate))
  librosa.display.waveplot(speech, sr=rate)
  plt.show()
  print(f"ASR hypothesis: {text_normalizer(text)}")
  print("*" * 50)


s2t(file)
