import gc
import jiwer

def cer(predictions, references, chunk_size=None):
    if chunk_size is None:
        preds = [char for seq in predictions for char in list(seq)]
        refs = [char for seq in references for char in list(seq)]
        return jiwer.wer(refs, preds)
    start = 0
    end = chunk_size
    H, S, D, I = 0, 0, 0, 0
    while start < len(references):
        preds = [char for seq in predictions[start:end]
                for char in list(seq)]
        refs = [char for seq in references[start:end]
                for char in list(seq)]
        chunk_metrics = jiwer.compute_measures(refs, preds)
        H = H + chunk_metrics["hits"]
        S = S + chunk_metrics["substitutions"]
        D = D + chunk_metrics["deletions"]
        I = I + chunk_metrics["insertions"]
        start += chunk_size
        end += chunk_size
        del preds
        del refs
        del chunk_metrics
        gc.collect()
    return float(S + D + I) / float(H + S + D)
