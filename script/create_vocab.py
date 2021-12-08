from transformers import Wav2Vec2CTCTokenizer
import json
import os
import re
import torch
import optuna
from datetime import date
import numpy as np

from helper.dataCollatorCTCWithPadding import DataCollatorCTCWithPadding
from transformers import Trainer, TrainingArguments
from datasets import concatenate_datasets

from helper.parser import parse_arguments
from helper.get_model import get_model

from metrics.wer import wer

from import_ds.import_dataset import import_dataset

hu = import_dataset('hu')
torgo = import_dataset('torgo')

hu = hu.cast(torgo.features)
ds = concatenate_datasets([hu, torgo])

chars_to_ignore_regex = '[\,\?\.\!\-\;\:\"]'

def remove_special_characters(batch):
    batch["target"] = re.sub(
        chars_to_ignore_regex, '', batch["target"]).lower()
    return batch

ds = ds.map(remove_special_characters)

def extract_all_chars(batch):
    all_text = " ".join(batch["target"])
    vocab = list(set(all_text))
    return {"vocab": [vocab], "all_text": [all_text]}

vocabs = ds.map(extract_all_chars, batched=True, batch_size=-1, keep_in_memory=True) 

vocab_list = list(set(vocabs["vocab"][0]))

vocab_dict = {v: k for k, v in enumerate(vocab_list)}

vocab_dict["|"] = vocab_dict[" "]
del vocab_dict[" "]

vocab_dict["[UNK]"] = len(vocab_dict)
vocab_dict["[PAD]"] = len(vocab_dict)
print(len(vocab_dict))

with open('/work/herzig/datasets/vocab.json', 'w') as vocab_file:
    json.dump(vocab_dict, vocab_file)


tokenizer = Wav2Vec2CTCTokenizer(
    "/work/herzig/datasets/vocab.json", unk_token="[UNK]", pad_token="[PAD]", word_delimiter_token="|")
