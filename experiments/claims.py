"""Claim extraction: split a generated report into sentences and label each with CheXbert.

A finding is claimed when CheXbert marks it positive in at least one sentence. Uncertain mentions
("possible effusion") are kept separately so they can be counted either way.
"""
import os, re
from common import CHEXBERT, CHEXBERT_TO_F

CACHE = os.path.expanduser("~/.cache/chexbert")


def sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if len(s.strip()) > 2]


class Labeler:
    def __init__(self, device="cpu"):
        import torch, f1chexbert.f1chexbert as fc
        from huggingface_hub import hf_hub_download
        os.makedirs(CACHE, exist_ok=True)
        fc.CACHE_DIR = CACHE
        if not os.path.exists(f"{CACHE}/chexbert.pth"):       # the package's own downloader uses a removed hub argument
            hf_hub_download("StanfordAIMI/RRG_scorers", "chexbert.pth", local_dir=CACHE)
        self.m = fc.F1CheXbert(device=device)

    def label(self, sent):
        """CheXbert classification mode: 1 positive, 0 negative, -1 uncertain, '' not mentioned."""
        return dict(zip(CHEXBERT, self.m.get_label(sent, mode="classification")))

    def extract(self, text):
        """{finding: {"status": "pos"|"unc", "sent": index}} over the shared space (plus unverifiable ones)."""
        out = {}
        for i, s in enumerate(sentences(text)):
            for name, v in self.label(s).items():
                f = CHEXBERT_TO_F.get(name)
                if f is None or v in ("", 0):
                    continue
                st = "pos" if v == 1 else "unc"
                if f not in out or (out[f]["status"] == "unc" and st == "pos"):
                    out[f] = {"status": st, "sent": i}
        return out


def sentence_logprob(text, tokens):
    """Mean token log-probability of each sentence of `text`, from the generation's (token, logprob) list."""
    sents, spans, pos = sentences(text), [], 0
    for s in sents:
        start = text.find(s, pos); spans.append((start, start + len(s))); pos = start + len(s)
    acc = [[] for _ in sents]
    joined = "".join(t for t, _ in tokens)
    if text[:20] not in joined:          # text was post-processed (e.g. box tags removed): use the report mean
        m = sum(lp for _, lp in tokens) / len(tokens) if tokens else None
        return [m] * len(sents)
    off = joined.find(text[:20]) if text[:20] in joined else 0     # leading whitespace may differ
    c = -off
    for t, lp in tokens:
        mid = c + len(t) / 2
        for j, (a, b) in enumerate(spans):
            if a <= mid < b:
                acc[j].append(lp); break
        c += len(t)
    return [sum(a) / len(a) if a else None for a in acc]


if __name__ == "__main__":
    assert sentences("No effusion. Mild cardiomegaly!\nClear lungs.") == ["No effusion.", "Mild cardiomegaly!", "Clear lungs."]
    toks = [("No", -0.1), (" effusion", -0.2), (".", -0.1), (" Big", -1.0), (" heart", -1.0), (".", -1.0)]
    lp = sentence_logprob("No effusion. Big heart.", toks)
    assert abs(lp[0] - (-0.4 / 3)) < 1e-9 and abs(lp[1] + 1.0) < 1e-9, lp
    print("claims.py ok")
