import torch

from helper.parser import parse_arguments
from helper.get_model import get_model
from metrics.wer import wer
from metrics.cer import cer
from import_ds.import_dataset import import_dataset


def main():
    args = parse_arguments()

    processor, model, device = get_model(args.l, args.m, args.local)  # Load tokenizer and model
    ds = import_dataset(args.d, args.local)  # Load a list of datasets

    def evaluate(batch):
        inputs = processor(batch['speech'], sampling_rate=16_000, return_tensors="pt", padding=True)

        with torch.no_grad():
            logits = model(inputs.input_values.to(device), attention_mask=inputs.attention_mask.to(device)).logits

        pred_ids = torch.argmax(logits, dim=-1)
        batch['pred_text'] = processor.batch_decode(pred_ids)
        return batch

    for dataset in ds:
        result = dataset.map(evaluate, batched=True, batch_size=8)

        predictions = [x.upper() for x in result['pred_text']]
        references = [x.upper() for x in result['target']]

        m_wer = wer(predictions=predictions, references=references, chunk_size=500) * 100
        m_cer = cer(predictions=predictions, references=references, chunk_size=500) * 100

        print('Model: ' + str(args.m) + ', patient: ' + dataset[0]['id'])
        print('WER: ' + str(m_wer))
        print('CER: ' + str(m_cer))


if __name__ == "__main__":
    main()
