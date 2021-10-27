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

###
### Following this guide https://huggingface.co/blog/fine-tune-xlsr-wav2vec2
###


def main():
    args = parse_arguments()

    tr_ds, te_ds = import_dataset(args.d, args.local, True)

    os.environ["WANDB_DISABLED"] = "true"
    
    def ft(train_ds, eval_ds, dir, t_args):

        processor, model, device = get_model(
            args.l, args.m, args.local)  # Load tokenizer and model

        # Freeze all layers except the last two
        if args.llo:
            for name, param in model.named_parameters():
                if not ('lm_head' in name):
                    param.requires_grad = False

        def compute_metrics(pred):
            pred_logits = pred.predictions
            pred_ids = np.argmax(pred_logits, axis=-1)

            pred.label_ids[pred.label_ids == -
                           100] = processor.tokenizer.pad_token_id

            pred_str = processor.batch_decode(pred_ids)
            # we do not want to group tokens when computing the metrics
            label_str = processor.batch_decode(
                pred.label_ids, group_tokens=False)

            m_wer = wer(predictions=pred_str,
                        references=label_str, chunk_size=500) * 100

            return {"wer": m_wer}

        def prep_dataset(batch):
            batch["input_values"] = processor(
                batch["speech"], sampling_rate=16_000).input_values

            with processor.as_target_processor():
                batch["labels"] = processor(batch["target"]).input_ids
            return batch

        training_args = TrainingArguments(
            output_dir=dir,
            group_by_length=True,
            per_device_train_batch_size=t_args['batch_size'],
            gradient_accumulation_steps=2,
            evaluation_strategy='steps',
            num_train_epochs=t_args['epoch'],
            fp16=True if not args.local else False,
            save_steps=400,
            eval_steps=400,
            logging_steps=400,
            learning_rate=t_args['learning_rate'],
            warmup_steps=500,
            save_total_limit=2,
        )

        eval_ds = eval_ds.map(prep_dataset, batched=True, batch_size=4).remove_columns(
            ['id', 'target', 'speech'])
        train_ds = train_ds.map(prep_dataset, batched=True, batch_size=4).remove_columns([
            'id', 'target', 'speech'])

        data_collator = DataCollatorCTCWithPadding(
            processor=processor, padding=True)

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

        print('---------------------------------------------------')
        print('Evaluation: ')

        return float(trainer.evaluate(batch_size=t_args['batch_size'])['eval_wer'])

    # def objective(trail):
    #     lr = trail.suggest_loguniform('learning_rate', 1e-5, 3e-4)
    #     bs = trail.suggest_int('batch_size', 8, 16, step=4)
    #     ep = trail.suggest_int('epoch', 10, 30, step=10)

    #     t_args = {'learning_rate': lr, 'batch_size': bs, 'epoch': ep}

    #     dir = '/home/tim/Documents/training/results/' + os.path.join(str(date.today()), str(args.d) + ('_llo' if args.llo else '_al')) if args.local else '/work/herzig/results/train/model/' + os.path.join(
    #         str(date.today()), str(args.d) + ('_llo' if args.llo else '_al'))
        
    #     if not os.path.exists(dir):
    #         os.makedirs(dir)
            
    #     return ft(tr_ds, te_ds, dir, t_args)

    # study = optuna.create_study(direction='minimize')
    # study.optimize(objective,  n_trials=10)

    # print('Best WER: ' + str(study.best_trial.value))

    dir = '/home/tim/Documents/training/results/' + os.path.join(str(date.today()), str(args.d) + ('_llo' if args.llo else '_al')) if args.local else '/work/herzig/results/train/model/' + os.path.join(
        str(date.today()), str(args.d) + ('_llo' if args.llo else '_al'))

    if not os.path.exists(dir):
        os.makedirs(dir)
    
    t_args = {'learning_rate': 1e-4, 'batch_size': 2, 'epoch': 30}

    wer = ft(tr_ds, te_ds, dir, t_args)

    print('Fine tune ended with WER = ' + str(wer))

if __name__ == "__main__":
    main()
