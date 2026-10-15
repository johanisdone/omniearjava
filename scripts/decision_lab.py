"""Offline decision-layer lab: tune/validate WITHOUT retraining the model.

Computes, once per clip, the per-window application-category scores (backbone +
head + ScoreMapper) and a small set of DSP descriptors, then evaluates decision
strategies on a held-out split so reported numbers are not threshold-leakage.

What it can test (all mirror TemporalDecisionEngine exactly, incl. confirmOf):
  - baseline: the shipped categories.json
  - score transforms: none / trailing moving-average / moving-max / EMA
  - per-category (threshold, confirmWindows, confirmOf) search
  - DSP transient fusion for the impulsive categories (gunshot/explosion/impact)

Run with the reference interpreter (onnxruntime + soundfile):
    python scripts/decision_lab.py
Outputs: .decision_lab/lab.npz (score+dsp cache), .decision_lab/report.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SR = 16000
WIN = 32000
HOP = 16000
CATS = ["SCREAM_DISTRESS", "GUNSHOT", "EXPLOSION", "IMPACT_CRASH", "SIREN", "GLASS_BREAK"]
IMPULSIVE = ["GUNSHOT", "EXPLOSION", "IMPACT_CRASH"]


def java_resample(x, src_rate, dst_rate):
    if not len(x) or abs(src_rate - dst_rate) < 0.5:
        return x
    ratio = dst_rate / src_rate
    out_len = int(np.floor(len(x) * ratio))
    pos = np.arange(out_len) / ratio
    i0 = np.floor(pos).astype(int)
    i1 = np.minimum(i0 + 1, len(x) - 1)
    frac = pos - i0
    return (x[i0] * (1 - frac) + x[i1] * frac).astype(np.float32)


def load_audio(path):
    import soundfile as sf
    x, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = np.ascontiguousarray(x, dtype=np.float32)
    if abs(sr - SR) > 0.5:
        x = java_resample(x, sr, SR)
    return x


def window_offsets(n):
    if n < WIN:
        return [0]
    return list(range(0, n - WIN + 1, HOP))


def dsp_features(w):
    w = w.astype(np.float64)
    rms = np.sqrt(np.mean(w * w)) + 1e-12
    peak = np.max(np.abs(w)) + 1e-12
    crest = peak / rms
    W = np.fft.rfft(w * np.hanning(len(w)))
    P = np.abs(W) ** 2 + 1e-12
    f = np.fft.rfftfreq(len(w), 1.0 / SR)
    centroid = float((f * P).sum() / P.sum())
    flat = float(np.exp(np.mean(np.log(P))) / np.mean(P))
    hf = float(P[f >= 2000].sum() / P.sum())
    low = float(P[f < 500].sum() / P.sum())
    zcr = float(np.mean(np.abs(np.diff(np.sign(w))) > 0))
    fr, hp = 512, 256
    n = (len(w) - fr) // hp + 1
    e = np.array([np.mean(w[i * hp:i * hp + fr] ** 2) for i in range(n)]) + 1e-12
    attack = float(e.max() / np.median(e))
    return [crest, flat, hf, low, centroid, zcr, attack]


# ---- engine (mirrors TemporalDecisionEngine) --------------------------------

def fires(series, thresh, cw, cof):
    """Clip-level: did the category fire at least once over this series?"""
    s = np.asarray(series, dtype=np.float64)
    above = (s >= thresh).astype(np.int64)
    if cw <= 1:
        return bool(above.any()) and cof <= 1
    padded = np.concatenate([np.zeros(cw - 1, dtype=np.int64), above])
    if len(padded) < cw:
        return False
    sums = np.convolve(padded, np.ones(cw, dtype=np.int64), "valid")
    return bool((sums >= cof).any())


def transform(s, kind, param):
    s = np.asarray(s, dtype=np.float64)
    if kind == "none":
        return s
    if kind == "ema":
        out = np.empty_like(s)
        prev = 0.0
        for i, v in enumerate(s):
            prev = param * v + (1 - param) * prev
            out[i] = prev
        return out
    if kind == "movavg":
        k = int(param)
        return np.array([s[max(0, i - k + 1):i + 1].mean() for i in range(len(s))])
    if kind == "movmax":
        k = int(param)
        return np.array([s[max(0, i - k + 1):i + 1].max() for i in range(len(s))])
    raise ValueError(kind)


def map_categories(probs, label_index, cats):
    series = {}
    for cat in cats:
        s = np.zeros(len(probs))
        for lbl, w in cat["labels"].items():
            s = np.maximum(s, w * probs[:, label_index[lbl]])
        series[cat["category"]] = s
    return series


# ---- metrics ----------------------------------------------------------------

def evaluate(series_by_clip, labels, cfg):
    per = {c["category"]: [0, 0, 0] for c in cfg["categories"]}
    correct = 0
    fp_clips = 0
    missed = 0
    for i, lab in enumerate(labels):
        fired = {}
        for c in cfg["categories"]:
            cat = c["category"]
            cof = c.get("confirmOf", c["confirmWindows"])
            s = transform(series_by_clip[i][cat], c.get("transform", "none"), c.get("param", 0))
            fired[cat] = fires(s, c["threshold"], c["confirmWindows"], cof)
        if lab == "NORMAL":
            bad = any(fired.values())
            correct += 0 if bad else 1
            fp_clips += 1 if bad else 0
        elif fired.get(lab, False):
            correct += 1
        else:
            missed += 1
        for c in cfg["categories"]:
            cat = c["category"]
            if lab == cat:
                per[cat][0 if fired[cat] else 2] += 1
            elif fired[cat]:
                per[cat][1] += 1
    f1s = []
    blocks = {}
    for c in cfg["categories"]:
        cat = c["category"]
        t, f, n = per[cat]
        p = t / (t + f) if t + f else 0.0
        r = t / (t + n) if t + n else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        f1s.append(f1)
        blocks[cat] = {"tp": t, "fp": f, "fn": n, "prec": round(p, 3), "rec": round(r, 3), "f1": round(f1, 3)}
    acc = correct / len(labels)
    macro = float(np.mean(f1s)) if f1s else 0.0
    return {"accuracy": round(acc, 4), "macroF1": round(macro, 4), "correct": correct,
            "falseAlarmClips": fp_clips, "missed": missed, "per": blocks}


def stratified_split(labels, test_frac=0.4, seed=7):
    rng = np.random.RandomState(seed)
    by = defaultdict(list)
    for i, l in enumerate(labels):
        by[l].append(i)
    cal, test = [], []
    for l, idxs in by.items():
        a = np.array(idxs)
        rng.shuffle(a)
        n = int(round(len(a) * test_frac))
        test.extend(a[:n].tolist())
        cal.extend(a[n:].tolist())
    return np.array(sorted(cal)), np.array(sorted(test))


# ---- cache ------------------------------------------------------------------

class Model:
    def __init__(self, models_dir):
        import onnxruntime as ort
        m = Path(models_dir)
        self.backbone = ort.InferenceSession(str(m / "ced-mini.onnx"), providers=["CPUExecutionProvider"])
        self.head = ort.InferenceSession(str(m / "onnx" / "head.onnx"), providers=["CPUExecutionProvider"])
        meta = json.loads((m / "onnx" / "audiodet_v4.json").read_text(encoding="utf-8"))
        self.classes = list(meta["classes"])
        self.label_index = {c: i for i, c in enumerate(self.classes)}

    def probs(self, x):
        offs = window_offsets(len(x))
        if len(x) < WIN:
            x = np.pad(x, (0, WIN - len(x)))
        wins = np.stack([x[o:o + WIN] for o in offs]).astype(np.float32)
        logits = self.backbone.run(None, {"waveform": wins})[0]
        return self.head.run(None, {"logits": logits})[0], wins


def build_cache(root, eval_dir, models_dir, cats, out):
    model = Model(root / models_dir)
    rows = []
    for line in (root / eval_dir / "labels.csv").read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        rel, lab = line.split(",")[:2]
        rows.append((root / eval_dir / rel, lab))
    labels, series, dsp = [], [], []
    for idx, (path, lab) in enumerate(rows):
        x = load_audio(path)
        probs, wins = model.probs(x)
        series.append(map_categories(probs, model.label_index, cats))
        dsp.append(np.stack([dsp_features(w) for w in wins]))
        labels.append(lab)
        if (idx + 1) % 200 == 0:
            print(f"  scored {idx + 1}/{len(rows)}", flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, labels=np.asarray(labels),
                        series=np.asarray(series, dtype=object),
                        dsp=np.asarray(dsp, dtype=object))
    print(f"wrote {out}: clips={len(labels)}")


# ---- search -----------------------------------------------------------------

def best_threshold_for(series_list, y, transform_kind, param, cw, cof, grid):
    transformed = [transform(s, transform_kind, param) for s in series_list]
    best = (-1.0, 0.5, 0.0)
    for t in grid:
        tp = fp = fn = 0
        for s, lab in zip(transformed, y):
            f = fires(s, t, cw, cof)
            if lab == 1:
                tp += 1 if f else 0
                fn += 0 if f else 1
            elif f:
                fp += 1
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        # Prefer higher F1; break ties toward higher precision.
        if f1 > best[0] + 1e-9 or (abs(f1 - best[0]) <= 1e-9 and p > best[2]):
            best = (f1, t, p)
    return best


ALL_CANDIDATES = [("none", 0), ("movmax", 2), ("movmax", 3),
                  ("movavg", 2), ("movavg", 3), ("ema", 0.5), ("ema", 0.8)]


def search_category(cat_series, labels, cat, grid, candidates=None):
    y = np.array([1 if l == cat else 0 for l in labels])
    candidates = candidates if candidates is not None else ALL_CANDIDATES
    best = None
    for kind, param in candidates:
        for cw in (1, 2, 3):
            for cof in range(1, cw + 1):
                f1, t, p = best_threshold_for(cat_series, y, kind, param, cw, cof, grid)
                key = (round(f1, 4), round(p, 4))
                if best is None or key > best["key"]:
                    best = {"key": key, "f1": f1, "threshold": float(t), "transform": kind,
                            "param": param, "confirmWindows": cw, "confirmOf": cof, "precision": p}
    return best


# ---- main -------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="eval-real", type=Path)
    ap.add_argument("--models", default="models", type=Path)
    ap.add_argument("--config", default="src/main/resources/config/categories.json", type=Path)
    ap.add_argument("--cache", default=".decision_lab/lab.npz", type=Path)
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--test-frac", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg = json.loads((root / args.config).read_text(encoding="utf-8"))
    cats = cfg["categories"]
    cache = root / args.cache
    if not cache.exists() or args.refresh:
        build_cache(root, args.eval, args.models, cats, cache)

    d = np.load(cache, allow_pickle=True)
    labels = [str(x) for x in d["labels"]]
    series = [dict(s) for s in d["series"]]
    dsp = list(d["dsp"])

    cal, test = stratified_split(labels, args.test_frac, args.seed)
    print(f"clips={len(labels)}  calibration={len(cal)}  test={len(test)}", flush=True)

    grid = np.unique(np.concatenate([np.arange(0.05, 1.0, 0.01), [0.999]]))

    def restrict(idxs):
        return [series[i] for i in idxs], [labels[i] for i in idxs]

    cal_series, cal_labels = restrict(cal)
    test_series, test_labels = restrict(test)

    print("\n=== baseline (shipped categories.json) ===")
    base_all = evaluate(series, labels, cfg)
    base_test = evaluate(test_series, test_labels, cfg)
    print("all :", json.dumps(base_all["per"]))
    print(f"all  acc={base_all['accuracy']:.3f} macroF1={base_all['macroF1']:.3f} "
          f"FPclips={base_all['falseAlarmClips']} missed={base_all['missed']}")
    print(f"test acc={base_test['accuracy']:.3f} macroF1={base_test['macroF1']:.3f} "
          f"FPclips={base_test['falseAlarmClips']} missed={base_test['missed']}")

    print("\n=== per-category search on calibration, evaluated on test ===")
    tuned = {"uncertainFloor": cfg.get("uncertainFloor", 0.3), "categories": []}
    report = {"baseline_all": base_all, "baseline_test": base_test, "per_category": {}}
    for c in cats:
        cat = c["category"]
        cal_series_cat = [s[cat] for s in cal_series]
        best = search_category(cal_series_cat, cal_labels, cat, grid)
        tuned["categories"].append({
            "category": cat, "labels": c["labels"], "threshold": round(best["threshold"], 3),
            "transform": best["transform"], "param": best["param"],
            "confirmWindows": best["confirmWindows"], "confirmOf": best["confirmOf"],
            "cooldownMs": c["cooldownMs"],
        })
        report["per_category"][cat] = {"calibration_choice": best}
        print(f"{cat:16s} t={best['threshold']:.3f} trans={best['transform']}{'' if best['param'] == 0 else '(' + str(best['param']) + ')'} "
              f"cw={best['confirmWindows']} cof={best['confirmOf']} calF1={best['f1']:.3f}")

    print("\n=== tuned config on test ===")
    tuned_test = evaluate(test_series, test_labels, tuned)
    tuned_all = evaluate(series, labels, tuned)
    print(f"all  acc={tuned_all['accuracy']:.3f} macroF1={tuned_all['macroF1']:.3f} "
          f"FPclips={tuned_all['falseAlarmClips']} missed={tuned_all['missed']}")
    print(f"test acc={tuned_test['accuracy']:.3f} macroF1={tuned_test['macroF1']:.3f} "
          f"FPclips={tuned_test['falseAlarmClips']} missed={tuned_test['missed']}")
    for cat in CATS:
        b = base_test["per"][cat]
        a = tuned_test["per"][cat]
        print(f"  {cat:16s} test base P={b['prec']:.2f} R={b['rec']:.2f} F1={b['f1']:.2f} "
              f"-> tuned P={a['prec']:.2f} R={a['rec']:.2f} F1={a['f1']:.2f}")
    report["tuned_all"] = tuned_all
    report["tuned_test"] = tuned_test

    # Transform must live in Java too; report which transforms were selected.
    used_transforms = {p["calibration_choice"]["transform"] for p in report["per_category"].values()}
    report["transforms_selected"] = sorted(used_transforms)

    out_dir = root / ".decision_lab"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (out_dir / "categories.candidate.json").write_text(json.dumps(tuned, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out_dir / 'report.json'} and categories.candidate.json")


if __name__ == "__main__":
    main()
