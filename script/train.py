import torch

from helper.dataCollatorCTCWithPadding import DataCollatorCTCWithPadding
from transformers import Trainer, TrainingArguments
from datasets import concatenate_datasets

from helper.parser import parse_arguments
from helper.get_model import get_model

from metrics.compute_metrics import compute_metrics
from metrics.wer import wer
from metrics.cer import cer

from import_ds.import_dataset import import_dataset



def main():
    args = parse_arguments()

    processor, model, device = get_model(args.l, args.m, args.local)  # Load tokenizer and model
    ds = import_dataset(args.d, args.local)  # Load a list of datasets

    training_args = TrainingArguments(
        output_dir='/home/tim/Documents/training' if args.local else '/work/herzig/results/train',  # output directory
        group_by_length=True,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=2,
        evaluation_strategy='steps',
        num_train_epochs=30,
        fp16=True if not args.local else False,
        save_steps=400,
        eval_steps=400,
        logging_steps=400,
        learning_rate=3e-4,
        warmup_steps=500,
        save_total_limit=2,
    )

    def prep_dataset(batch):
        batch["input_values"] = processor(batch["speech"], sampling_rate=16000).input_values

        with processor.as_target_processor():
            batch["labels"] = processor(batch["target"]).input_ids
        return batch

    def ft(train_ds, eval_ds):
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

    def leave_one_out_evaluation(speaker_datasets):
        for speaker_dataset in speaker_datasets:
            speaker_datasets_wo_cur_speaker = speaker_datasets[:]
            speaker_datasets_wo_cur_speaker.remove(speaker_dataset)
            ds_wo_cur_speaker = concatenate_datasets(speaker_datasets_wo_cur_speaker)

            ft(ds_wo_cur_speaker, speaker_dataset)

    leave_one_out_evaluation(ds)


if __name__ == "__main__":
    main()
