import jiwer

def wer(predictions, references, chunk_size=None):
    if chunk_size is None:
            return jiwer.wer(references, predictions)
    start = 0
    end = chunk_size
    H, S, D, I = 0, 0, 0, 0
    while start < len(references):
        chunk_metrics = jiwer.compute_measures(
            references[start:end], predictions[start:end])
        H = H + chunk_metrics["hits"]
        S = S + chunk_metrics["substitutions"]
        D = D + chunk_metrics["deletions"]
        I = I + chunk_metrics["insertions"]
        start += chunk_size
        end += chunk_size
    return float(S + D + I) / float(H + S + D)
