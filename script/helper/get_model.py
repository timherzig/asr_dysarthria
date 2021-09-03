from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer


def get_model(lang, model, local):
    LANG_ID: str = lang
    MODEL_ID: str = model
    DEVICE: str = 'cpu'
    if not local:
        DEVICE: str = 'cuda'

    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)

    # if 'xlsr' in MODEL_ID:
    #    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)
    #    tokenizer = processor.tokenizer
    # else:
    #    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(MODEL_ID)

    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID)
    model.to(DEVICE)

    return processor, model, DEVICE
