import os
import re
import torch
import optuna
from datetime import date
import numpy as np

from helper.dataCollatorCTCWithPadding import DataCollatorCTCWithPadding
from transformers import Trainer, TrainingArguments, AdamW, get_scheduler
from datasets import concatenate_datasets, load_metric

from helper.parser import parse_arguments
from helper.get_model import get_model

from tqdm.auto import tqdm

from torch.utils.data import DataLoader

# from metrics.wer import wer

from import_ds.import_dataset import import_dataset

###
### Following this guide https://huggingface.co/blog/fine-tune-xlsr-wav2vec2
###


def main():
    args = parse_arguments()
    train_ds, test_ds = import_dataset(args.d, args.local, True)

    processor, model, device = get_model(args.l, args.m, args.local)
    tokenizer = processor.tokenizer

    def tokenize_function(examples):
        return tokenizer(examples['speech'], padding='max_length', truncation=True)

    train_ds = train_ds.map(tokenize_function, batched=True)
    train_ds = train_ds.remove_columns(['speech', 'id'])
    train_ds = train_ds.rename_column('target', 'labels')
    train_ds.set_format('torch')

    test_ds = test_ds.map(tokenize_function, batched=True)
    test_ds = test_ds.remove_columns(['speech', 'id'])
    test_ds = test_ds.rename_column('target', 'labels')
    test_ds.set_format('torch')
    
    train_dataloader = DataLoader(train_ds, shuffle=True, batch_size=8)
    test_dataloader = DataLoader(test_ds, batch_size=8)

    optimizer = AdamW(model.parameters(), lr=1e-4)

    num_epochs = 2
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    progress_bar = tqdm(range(num_training_steps))


    model.train()
    for epoch in range(num_epochs):
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)

    metric = load_metric('wer')

    model.eval()
    for batch in test_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        metric.add_batch(predictions=predictions, references=batch["labels"])

    metric.compute()


if __name__ == "__main__":
    main()
