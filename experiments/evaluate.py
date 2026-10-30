"""Score every run file under $CXR_ROOT/runs and write results/summary.csv (+ per-run JSON).

Metrics (definitions in the paper, Section 4):
  ESP / UCR      share of verifiable positive claims supported by expert evidence (text level, tau=0) and with the
                 model's own box overlapping an expert box (IoU >= 0.3, 0.5) or pointing inside one
  ER             share of expert-evidenced findings that the model claims (and localises, at tau>0)
  grounding      oracle phrase grounding on evidenced findings: mIoU, Acc@0.3/0.5, pointing game
  calibration    per-claim confidence signals vs. text-level support: ECE, Brier, AUROC, AURC, risk@80/90% coverage
  conformal      claim filter calibrated on chestxdet/clean of the same model; realised coverage elsewhere
  probes         label sites: AUROC / accuracy / ECE of P(abnormal)=1-P(normal) and P(TB)
  quality        artefact-awareness: report mentions image quality; mean P(quality adequate)
  reference      CheXbert micro-F1 (14 classes) against the reference report, where one exists
"""
import glob, json, os, re, sys
import numpy as np
from common import RUNS, ROOT, FINDINGS, iou

OUT = f"{ROOT}/results"
TAUS = (0.3, 0.5)
QUALITY_RE = re.compile(r"quality|motion|blur|nois|artifact|artefact|rotat|under-?exposed|over-?exposed|"
                        r"suboptimal|limited (study|evaluation|assessment)|grain|compression|poorly visualized", re.I)


def load(path):
    return [json.loads(l) for l in open(path)]


def center_in(b, e):
    cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return e[0] <= cx <= e[2] and e[1] <= cy <= e[3]


def best_iou(pred, ev):
    return max((iou(p, e) for p in pred for e in ev), default=0.0)


def hit(pred, ev):
    return any(center_in(p, e) for p in pred for e in ev)


# ---- claim table ------------------------------------------------------------------------------------------
def claim_rows(recs):
    """One row per verifiable positive greedy claim: support flags and the four confidence signals."""
    rows = []
    for r in recs:
        ver = set(r.get("verifiable", FINDINGS)) if r.get("has_boxes") else set()
        for f, c in r["claims"].items():
            if c["status"] != "pos" or f not in ver:
                continue
            ev = r["evidence"].get(f, [])
            locs = r["loc"].get(f, [])
            g = locs[0] if locs else []
            k = len(r["samples"])
            nonempty = [l[0] for l in locs if l]
            pair = [iou(a, b) for i, a in enumerate(nonempty) for b in nonempty[i + 1:]]
            rows.append(dict(
                id=r["id"], f=f, text=bool(ev),
                **{f"iou{t}": bool(ev) and best_iou(g, ev) >= t for t in TAUS},
                point=bool(ev) and hit(g, ev),
                conf_logprob=float(np.exp(c["logprob"])) if c.get("logprob") is not None else np.nan,
                conf_consistency=(sum(sc.get(f) == "pos" for sc in r["sample_claims"]) / k) if k else np.nan,
                conf_pyes=r["pyes"].get(f, np.nan),
                conf_evidence=(len(nonempty) / len(locs)) * (np.mean(pair) if pair else (1.0 if nonempty else 0.0))
                if locs else np.nan))
        for row in rows:
            if row["id"] == r["id"]:      # self-grounding score: mean of the image-conditioned signals
                row["conf_sgf"] = float(np.nanmean([row["conf_consistency"], row["conf_pyes"], row["conf_evidence"]]))
    return rows


def esp_block(rows, recs, prefix=""):
    out = {}
    n = len(rows)
    out["n_claims"] = n
    for key in ["text", *[f"iou{t}" for t in TAUS], "point"]:
        out[f"ESP_{key}"] = float(np.mean([r[key] for r in rows])) if n else np.nan
    out["UCR_text"] = 1 - out["ESP_text"] if n else np.nan
    # evidence recall
    tot, rec = 0, {k: 0 for k in ["text", *[f"iou{t}" for t in TAUS]]}
    for r in recs:
        if not r.get("has_boxes"):
            continue
        for f, ev in r["evidence"].items():
            if f not in FINDINGS or not ev:
                continue
            tot += 1
            c = r["claims"].get(f)
            if c and c["status"] == "pos":
                rec["text"] += 1
                g = (r["loc"].get(f) or [[]])[0]
                for t in TAUS:
                    rec[f"iou{t}"] += best_iou(g, ev) >= t
    for k, v in rec.items():
        out[f"ER_{k}"] = v / tot if tot else np.nan
    out["n_evidenced"] = tot
    # unsupported claims per image
    imgs = [r for r in recs if r.get("has_boxes")]
    out["unsupported_per_image"] = (n - sum(r["text"] for r in rows)) / len(imgs) if imgs else np.nan
    return out


def bootstrap_ci(rows, key, n=1000, seed=0):
    ids = sorted({r["id"] for r in rows})
    if not ids:
        return (np.nan, np.nan)
    by = {}
    for r in rows:
        by.setdefault(r["id"], []).append(r[key])
    rng, vals = np.random.default_rng(seed), []
    for _ in range(n):
        s = [v for i in rng.choice(ids, len(ids)) for v in by[i]]
        vals.append(np.mean(s))
    return tuple(np.percentile(vals, [2.5, 97.5]).round(4))


# ---- grounding ---------------------------------------------------------------------------------------------
def grounding(recs):
    ious, hits = [], []
    for r in recs:
        for f, ev in r["evidence"].items():
            if f in FINDINGS and ev and r["loc"].get(f):
                g = r["loc"][f][0]
                ious.append(best_iou(g, ev)); hits.append(hit(g, ev))
    if not ious:
        return {}
    a = np.array(ious)
    return dict(n_ground=len(a), mIoU=float(a.mean()), acc03=float((a >= 0.3).mean()), acc05=float((a >= 0.5).mean()),
                pointing=float(np.mean(hits)), no_box=float(np.mean([x == 0 for x in a])))


# ---- calibration -------------------------------------------------------------------------------------------
def ece(p, y, bins=15):
    edges, e = np.linspace(0, 1, bins + 1), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & ((p < hi) if hi < 1 else (p <= hi))
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def auroc(p, y):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else np.nan


def risk_coverage(p, y):
    """AURC and selective risk (error rate of kept items) at 80% / 90% coverage, keeping the most confident."""
    o = np.argsort(-p)
    err = 1 - y[o]
    risk = np.cumsum(err) / np.arange(1, len(y) + 1)
    at = lambda c: float(risk[max(int(np.ceil(c * len(y))) - 1, 0)])
    return float(risk.mean()), at(0.8), at(0.9)


def calib(p, y):
    m = ~np.isnan(p)
    p, y = np.clip(p[m], 0, 1), y[m].astype(float)
    if len(p) < 10:
        return {}
    aurc, r80, r90 = risk_coverage(p, y)
    return dict(n=int(len(p)), acc=float(y.mean()), ECE=ece(p, y), Brier=float(np.mean((p - y) ** 2)),
                AUROC=auroc(p, y), AURC=aurc, risk80=r80, risk90=r90)


SIGNALS = ["conf_logprob", "conf_consistency", "conf_pyes", "conf_evidence", "conf_sgf"]


# ---- conformal claim filter (Conformal Factuality, Mohri & Hashimoto 2024) ---------------------------------
def report_thresholds(rows, signal):
    """Per image: the smallest score threshold above which every kept claim is supported."""
    by = {}
    for r in rows:
        s = r[signal]
        by.setdefault(r["id"], []).append((-np.inf if np.isnan(s) else s, r["text"]))
    return {i: max([s for s, ok in cs if not ok], default=-np.inf) for i, cs in by.items()}, by


def conformal_q(rows, signal, alpha, n_images):
    t, _ = report_thresholds(rows, signal)
    ts = sorted(list(t.values()) + [-np.inf] * (n_images - len(t)))   # images without claims need no threshold
    k = int(np.ceil((1 - alpha) * (len(ts) + 1))) - 1
    return ts[min(k, len(ts) - 1)] if ts else np.inf


def conformal_eval(rows, signal, q, n_images):
    _, by = report_thresholds(rows, signal)
    ok = sum(all(okc for s, okc in cs if s > q) for cs in by.values()) + (n_images - len(by))
    kept = sum(s > q for cs in by.values() for s, _ in cs)
    return ok / n_images if n_images else np.nan, kept / max(1, sum(len(cs) for cs in by.values()))


def local_recalibration(rows, recs, signal, alpha=0.1, ms=(10, 25, 50), reps=200, seed=0):
    """Site-adaptive filter: set the threshold from m labelled images of the target site, test on the rest."""
    ids = sorted({r["id"] for r in recs if r.get("has_boxes")})
    out = {}
    rng = np.random.default_rng(seed)
    for m in ms:
        if len(ids) < m + 20:
            continue
        cov, kept = [], []
        for _ in range(reps):
            cal = set(rng.choice(ids, m, replace=False))
            rc = [r for r in rows if r["id"] in cal]; rt = [r for r in rows if r["id"] not in cal]
            qq = conformal_q(rc, signal, alpha, m)
            c, k = conformal_eval(rt, signal, qq, len(ids) - m)
            cov.append(c); kept.append(k)
        out[f"local{m}_{signal}_a{alpha}_coverage"] = float(np.mean(cov))
        out[f"local{m}_{signal}_a{alpha}_kept"] = float(np.mean(kept))
    return out


# ---- evidence ablation -------------------------------------------------------------------------------------
def ablation(clean, ev, rnd, n_boot=1000, seed=0):
    """For expert-evidenced findings: how often a clean-image claim (and P(yes)) survives removing the evidence,
    against removing an equal area elsewhere. Evidence dependence = drop(evidence) - drop(random)."""
    E, R = {r["id"]: r for r in ev}, {r["id"]: r for r in rnd}
    items = []
    for c in clean:
        if c["id"] not in E or c["id"] not in R:
            continue
        if len(R[c["id"]].get("masked") or []) < len(E[c["id"]].get("masked") or []):
            continue                     # control could not remove an equal area: not a matched comparison
        for f, boxes in c["evidence"].items():
            if f not in FINDINGS or not boxes:
                continue
            claimed = c["claims"].get(f, {}).get("status") == "pos"
            items.append(dict(id=c["id"], f=f, claimed=claimed,
                              keep_ev=E[c["id"]]["claims"].get(f, {}).get("status") == "pos",
                              keep_rnd=R[c["id"]]["claims"].get(f, {}).get("status") == "pos",
                              dp_ev=c["pyes"][f] - E[c["id"]]["pyes"][f], dp_rnd=c["pyes"][f] - R[c["id"]]["pyes"][f]))
    if not items:
        return {}
    cl = [i for i in items if i["claimed"]]
    out = dict(abl_n_items=len(items), abl_n_claimed=len(cl),
               abl_claim_drop_ev=float(np.mean([not i["keep_ev"] for i in cl])) if cl else np.nan,
               abl_claim_drop_rnd=float(np.mean([not i["keep_rnd"] for i in cl])) if cl else np.nan,
               abl_dpyes_ev=float(np.mean([i["dp_ev"] for i in items])),
               abl_dpyes_rnd=float(np.mean([i["dp_rnd"] for i in items])))
    out["abl_EDI_claims"] = out["abl_claim_drop_ev"] - out["abl_claim_drop_rnd"]
    out["abl_EDI_pyes"] = out["abl_dpyes_ev"] - out["abl_dpyes_rnd"]
    ids = sorted({i["id"] for i in items}); by = {}
    for i in items:
        by.setdefault(i["id"], []).append(i)
    rng, vals = np.random.default_rng(seed), []
    for _ in range(n_boot):
        s = [i for k in rng.choice(ids, len(ids)) for i in by[k]]
        vals.append(np.mean([i["dp_ev"] - i["dp_rnd"] for i in s]))
    out["abl_EDI_pyes_CI"] = tuple(np.percentile(vals, [2.5, 97.5]).round(4))
    return out


# ---- label-site probes -------------------------------------------------------------------------------------
def probes(recs):
    out = {}
    for key, score in (("abnormal", lambda r: 1 - r["pyes"]["normal"]), ("tb", lambda r: r["pyes"]["tb"])):
        rs = [r for r in recs if key in r.get("labels", {})]
        if len(rs) < 10:
            continue
        p, y = np.array([score(r) for r in rs]), np.array([r["labels"][key] for r in rs], float)
        conf = np.where(p >= 0.5, p, 1 - p)
        cc = calib(conf, ((p >= 0.5) == (y == 1)).astype(float))
        out.update({f"{key}_AUROC": auroc(p, y), f"{key}_acc": float(((p >= 0.5) == (y == 1)).mean()),
                    f"{key}_ECE": cc.get("ECE", np.nan), f"{key}_risk80": cc.get("risk80", np.nan), f"{key}_n": len(rs)})
        claim_abn = np.array([any(c["status"] == "pos" for c in r["claims"].values()) for r in rs], float)
        out[f"{key}_claim_acc"] = float((claim_abn == y).mean())
    return out


# ---- reference CheXbert F1 ---------------------------------------------------------------------------------
_LAB = None


def chexbert_f1(recs):
    global _LAB
    rs = [r for r in recs if r.get("reference")]
    if not rs:
        return {}
    if _LAB is None:
        from claims import Labeler
        _LAB = Labeler("cpu")
    hyp = np.array([_LAB.m.get_label(r["report"])[:13] for r in rs], bool)
    ref = np.array([_LAB.m.get_label(r["reference"])[:13] for r in rs], bool)
    tp, fp, fn = (hyp & ref).sum(), (hyp & ~ref).sum(), (~hyp & ref).sum()
    return dict(n_ref=len(rs), chexbert_microF1=float(2 * tp / max(1, 2 * tp + fp + fn)))


def main():
    os.makedirs(OUT, exist_ok=True)
    files = sorted(glob.glob(f"{RUNS}/*/*/*.jsonl"))
    q = {}                                                      # conformal thresholds from chestxdet/clean
    rows_all, table = {}, []
    for p in files:
        model, site, cond = p.split("/")[-3], p.split("/")[-2], os.path.basename(p)[:-6]
        recs = load(p)
        rows = claim_rows(recs)
        rows_all[(model, site, cond)] = (rows, recs)
    cal_site = "chestxdet_cal" if any(k[1] == "chestxdet_cal" for k in rows_all) else "chestxdet"
    for (model, site, cond), (rows, recs) in rows_all.items():
        if site == cal_site and cond == "clean":
            n_img = sum(r.get("has_boxes", False) for r in recs)
            for s in SIGNALS:
                for a in (0.05, 0.1, 0.2):
                    q[(model, s, a)] = conformal_q(rows, s, a, n_img)
    for (model, site, cond), (rows, recs) in rows_all.items():
        res = dict(model=model, site=site, cond=cond, n_images=len(recs),
                   sec_per_image=float(np.mean([r["sec"] for r in recs])))
        res.update(esp_block(rows, recs))
        if rows:
            res["ESP_text_CI"] = bootstrap_ci(rows, "text")
        res.update(grounding(recs))
        y = np.array([r["text"] for r in rows], float)
        for s in SIGNALS:
            for k, v in calib(np.array([r[s] for r in rows], float), y).items():
                res[f"{s}__{k}"] = v
        n_img = sum(r.get("has_boxes", False) for r in recs)
        for s in SIGNALS:
            for a in (0.05, 0.1, 0.2):
                if (model, s, a) in q and n_img:
                    cov, kept = conformal_eval(rows, s, q[(model, s, a)], n_img)
                    res[f"conf_{s}_a{a}_coverage"], res[f"conf_{s}_a{a}_kept"] = cov, kept
        if n_img and site != cal_site:
            for s in ("conf_consistency", "conf_sgf"):
                res.update(local_recalibration(rows, recs, s))
        if cond == "clean" and (model, site, "evmask_1") in rows_all and (model, site, "randmask_1") in rows_all:
            res.update(ablation(recs, rows_all[(model, site, "evmask_1")][1], rows_all[(model, site, "randmask_1")][1]))
        res.update(probes(recs))
        res["quality_mention"] = float(np.mean([bool(QUALITY_RE.search(r["report"])) for r in recs]))
        res["p_quality_ok"] = float(np.mean([r["pyes"]["quality"] for r in recs]))
        res["claims_per_image"] = float(np.mean([sum(c["status"] == "pos" for c in r["claims"].values()) for r in recs]))
        res.update(chexbert_f1(recs))
        json.dump(res, open(f"{OUT}/{model}__{site}__{cond}.json", "w"), indent=1, default=str)
        table.append(res)
        print(model, site, cond, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res.items()
                                   if k in ("n_images", "ESP_text", "ESP_iou0.3", "ER_text", "mIoU", "abnormal_AUROC", "abl_EDI_claims", "abl_EDI_pyes")})
    import csv
    keys = sorted({k for r in table for k in r}, key=lambda k: (k not in ("model", "site", "cond"), k))
    with open(f"{OUT}/summary.csv", "w", newline="") as f:
        w = csv.DictWriter(f, keys); w.writeheader(); w.writerows(table)
    print("wrote", f"{OUT}/summary.csv", len(table), "rows")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        p, y = np.array([0.9, 0.8, 0.3, 0.2]), np.array([1, 1, 0, 0.])
        assert auroc(p, y) == 1.0 and risk_coverage(p, y)[1] == 0.5
        rows = [dict(id="a", text=True, s=0.9), dict(id="a", text=False, s=0.4), dict(id="b", text=True, s=0.7)]
        qq = conformal_q(rows, "s", 0.2, 3); assert qq == 0.4, qq
        assert conformal_eval(rows, "s", qq, 3) == (1.0, 2 / 3)
        print("evaluate.py ok")
    else:
        main()
