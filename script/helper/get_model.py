from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2CTCTokenizer


def get_model(lang, model, local):
    LANG_ID: str = lang
    MODEL_ID: str = model
    DEVICE: str = 'cpu'
    if not local:
        DEVICE: str = 'cuda'

    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID, force_download=True)
    model.to(DEVICE)
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID, force_download=True)

    return processor, model, DEVICE
