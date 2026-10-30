"""Run one model on one site under one image condition and write one JSON line per image.

  python generate.py --model qwen25vl7b --site chestxdet --cond clean
  python generate.py --model chexagent2 --site chestxdet --cond blur:2 --n 200

Output: $CXR_ROOT/runs/<model>/<site>/<cond>.jsonl (resumes: images already written are skipped).
Each line holds the greedy report with token log-probs, K sampled reports, the claims CheXbert finds in each,
P(yes) for every finding and for quality / normal / TB questions, and box lists (greedy + samples) for every
finding that was claimed or has expert evidence. Expert boxes are stored already mapped into the corrupted frame.
"""
import argparse, json, os, random, time
# every model is already cached; skip the Hub check so an internet outage does not kill the run
# (set HF_HUB_OFFLINE=0 when a new model, e.g. MedGemma, still has to be downloaded)
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from PIL import Image
from common import RUNS, FINDINGS, PROMPT_NAME, UNVERIFIABLE, corrupt, mask_evidence, load_manifest
from claims import Labeler, sentence_logprob
from models import MODELS

Q_FINDING = "Does this chest X-ray show {f}?"
Q_EXTRA = {"quality": "Is the image quality of this chest X-ray adequate for diagnostic interpretation?",
           "normal": "Is this chest X-ray normal?",
           "tb": "Does this chest X-ray show signs of tuberculosis?"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True); ap.add_argument("--site", required=True)
    ap.add_argument("--cond", default="clean"); ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--k", type=int, default=5); ap.add_argument("--kloc", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.7); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    kind, sev = (None, 0) if a.cond == "clean" else (a.cond.split(":")[0], int(a.cond.split(":")[1]))
    ablation = kind in ("evmask", "randmask")             # evidence ablation: only images with expert evidence
    recs = load_manifest(a.site)
    if ablation:   # images with evidence where the control can remove an equal area (same seed as below)
        def matched(r):
            n = len({tuple(b) for bs in r["evidence"].values() for b in bs})
            seed = a.seed + int.from_bytes(r["id"].encode()[-4:], "little")
            return n > 0 and len(mask_evidence(Image.open(r["path"]).convert("RGB"), r, "randmask", seed)[1]) == n
        recs = [r for r in recs if any(r.get("evidence", {}).values()) and matched(r)]
    if a.n and a.n < len(recs):
        recs = sorted(random.Random(a.seed).sample(recs, a.n), key=lambda r: r["id"])
    out = f"{RUNS}/{a.model}/{a.site}/{a.cond.replace(':', '_')}.jsonl"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    done = {json.loads(l)["id"] for l in open(out)} if os.path.exists(out) else set()
    todo = [r for r in recs if r["id"] not in done]
    print(f"{a.model} {a.site} {a.cond}: {len(todo)} to do, {len(done)} done", flush=True)
    if not todo:
        return
    import torch
    torch.manual_seed(a.seed)
    m, lab = MODELS[a.model](), Labeler("cpu")

    with open(out, "a") as fo:
        for i, r in enumerate(todo):
            t0 = time.time()
            img = Image.open(r["path"]).convert("RGB")
            seed = a.seed + int.from_bytes(r["id"].encode()[-4:], "little")
            masked = None
            if ablation:
                img, masked = mask_evidence(img, r, kind, seed=seed); bf = lambda b: b
            else:
                img, bf = corrupt(img, kind, sev, seed=seed)
            w, h = img.size
            rep = m.report(img)
            claims = lab.extract(rep["text"])
            slp = sentence_logprob(rep["text"], rep["tokens"])
            for c in claims.values():
                c["logprob"] = slp[c["sent"]] if c["sent"] < len(slp) else None
            samples = m.samples(img, a.k, a.temp) if a.k else []
            sample_claims = [{f: c["status"] for f, c in lab.extract(t).items()} for t in samples]
            qs = [Q_FINDING.format(f=PROMPT_NAME[f]) for f in FINDINGS] + list(Q_EXTRA.values())
            pyes = dict(zip(FINDINGS + list(Q_EXTRA), m.p_yes_many(img, qs)))

            evidence = {f: [[b[0] / w, b[1] / h, b[2] / w, b[3] / h] for b in map(bf, bs)]
                        for f, bs in r.get("evidence", {}).items()}
            loc, loc_raw = {}, {}
            if r.get("has_boxes") and a.kloc >= 0:          # --kloc -1 skips localisation
                targets = {f for f in claims if f not in UNVERIFIABLE} | {f for f in evidence if f in FINDINGS}
                fs = sorted(targets)
                res = m.locate_many(img, [PROMPT_NAME[f] for f in fs], k=a.kloc, temp=a.temp)
                for f in fs:
                    loc[f], loc_raw[f] = res[PROMPT_NAME[f]]

            fo.write(json.dumps(dict(
                id=r["id"], site=a.site, cond=a.cond, model=a.model, labels=r.get("labels", {}),
                evidence=evidence, has_boxes=r.get("has_boxes", False), reference=r.get("report"),
                report=rep["text"], report_raw=rep.get("raw"), report_boxes=rep.get("report_boxes"), tokens=rep["tokens"], claims=claims, samples=samples,
                sample_claims=sample_claims, pyes=pyes, loc=loc, loc_raw=loc_raw, masked=masked, sec=round(time.time() - t0, 2))) + "\n")
            fo.flush()
            if i % 10 == 0:
                print(f"[{i + 1}/{len(todo)}] {r['id']} {time.time() - t0:.1f}s claims={sorted(claims)}", flush=True)


if __name__ == "__main__":
    main()
