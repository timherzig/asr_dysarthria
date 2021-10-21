import torch
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC, Wav2Vec2FeatureExtractor, AutoTokenizer, AutoModelForPreTraining


def get_model(lang, model, local):
    LANG_ID: str = lang
    MODEL_ID: str = model
    DEVICE: str = 'cpu'
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    if MODEL_ID == ('facebook/wav2vec2-large-xlsr-53' or '/work/herzig/models/ml-fb-wav2vec2-large-xlsr-53/'):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        feature_extractor = Wav2Vec2FeatureExtractor(feature_size=1, sampling_rate=16000, padding_value=0.0, do_normalize=True, return_attention_mask=False)
        processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
        model = AutoModelForPreTraining.from_pretrained(MODEL_ID).to(device)
        return processor, model, device      

    model = Wav2Vec2ForCTC.from_pretrained(MODEL_ID).to(device)
    processor = Wav2Vec2Processor.from_pretrained(MODEL_ID)

    return processor, model, device
