import os
from optuna import trial
import torch
from datetime import date
import numpy as np

from helper.dataCollatorCTCWithPadding import DataCollatorCTCWithPadding
from transformers import Trainer, TrainingArguments
from datasets import concatenate_datasets

from helper.parser import parse_arguments
from helper.get_model import get_model

from metrics.wer import wer

from import_ds.import_dataset import import_dataset

###
### Following this guide https://huggingface.co/blog/fine-tune-xlsr-wav2vec2
###

def main():
    args = parse_arguments()

    processor, model, device = get_model(args.l, args.m, args.local)  # Load tokenizer and model
    ds = import_dataset(args.d, args.local)  # Load a list of datasets

    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)

        pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id

        pred_str = processor.batch_decode(pred_ids)
        # we do not want to group tokens when computing the metrics
        label_str = processor.batch_decode(pred.label_ids, group_tokens=False)

        m_wer = wer(predictions=pred_str, references=label_str)

        return {"wer": m_wer}


    def prep_dataset(batch):
        batch["input_values"] = processor(batch["speech"], sampling_rate=16000).input_values

        with processor.as_target_processor():
            batch["labels"] = processor(batch["target"]).input_ids
        return batch

    def ft(train_ds, eval_ds, dir, t_args):
        os.makedirs(dir)

        training_args = TrainingArguments(
           output_dir=dir,
           # output directory
           group_by_length=True,
           per_device_train_batch_size=t_args['batch_size'],
           gradient_accumulation_steps=2,
           evaluation_strategy='steps',
           num_train_epochs=30,
           fp16=True if not args.local else False,
           save_steps=400,
           eval_steps=400,
           logging_steps=400,
           learning_rate=t_args['learning_rate'],
           warmup_steps=0,
           save_total_limit=2,
        )


        eval_ds = eval_ds.map(prep_dataset, batched=True, batch_size=4).remove_columns(
            ['id', 'target', 'speech'])
        train_ds = train_ds.map(prep_dataset, batched=True, batch_size=4).remove_columns([
            'id', 'target', 'speech'])

        data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

        trainer = Trainer(
            model=model,
            data_collator=data_collator,
            args=training_args,
            compute_metrics=compute_metrics,
            train_dataset=train_ds,
            eval_dataset=eval_ds,
            tokenizer=processor.feature_extractor,
        )

        trainer.train()
        # GOAL: TODO: return wer of final trained model

    
    t_args= {'learning_rate': 3e-4, 'batch_size': 8}


    train_ds = 
    eval_ds = 


    ft(train_ds, eval_ds, t_args)


if __name__ == "__main__":
    main()
