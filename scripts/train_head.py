"""Train a replacement head (ced-mini logits -> 17 threat probabilities).

Consumes .head_train/features.npz from extract_features.py and produces
.head_train/head.npz (folded StandardScaler + linear/MLP weights) plus an
honest cross-validated report. Import via export_head_onnx.py.

The clip decision mirrors the Java TemporalDecisionEngine exactly: a category
fires iff confirmWindows CONSECUTIVE 2 s windows score >= threshold (a clip with
fewer windows than confirmWindows can never fire). Thresholds are tuned on
out-of-fold predictions so the reported accuracy is a generalisation estimate.

Several per-category confirm-window settings are compared (impulsive events are
short and need 1 window; sustained events benefit from 2-window hysteresis).

Run with the system interpreter (has scikit-learn/scipy):
    python scripts/train_head.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

CAT_TO_LABELS = {
    "SCREAM_DISTRESS": ["scream"],
    "GUNSHOT": ["gunshot"],
    "EXPLOSION": ["explosion"],
    "IMPACT_CRASH": ["vehicle_crash", "knock"],
    "SIREN": ["siren"],
    "GLASS_BREAK": ["glass"],
}
USED = ["scream", "gunshot", "explosion", "vehicle_crash", "knock", "siren", "glass"]
CATS = list(CAT_TO_LABELS)

MODELS = [
    {"name": "linear"},
    {"name": "mlp", "hidden": (256, 128), "seeds": (0,)},
    {"name": "mlp", "hidden": (512, 256), "seeds": (0,)},
    {"name": "mlp", "hidden": (256, 128), "seeds": (0, 1, 2)},
    {"name": "mlp", "hidden": (512, 256), "seeds": (0, 1, 2)},
]

CW_CONFIGS = {
    "all2": {c: 2 for c in CATS},
    "all1": {c: 1 for c in CATS},
    "impulsive1": {"SCREAM_DISTRESS": 2, "GUNSHOT": 1, "EXPLOSION": 1,
                   "IMPACT_CRASH": 1, "SIREN": 2, "GLASS_BREAK": 1},
    "short1": {"SCREAM_DISTRESS": 1, "GUNSHOT": 1, "EXPLOSION": 1,
               "IMPACT_CRASH": 1, "SIREN": 2, "GLASS_BREAK": 1},
}


def clip_score(series, cw):
    """Category clip score: max over windows i of min(s[i..i+cw-1]); clips with
    fewer than cw windows score 0.0 (all thresholds are > 0, so they never fire)."""
    s = np.asarray(series, dtype=np.float64)
    if len(s) < cw:
        return 0.0
    if cw == 1:
        return float(s.max())
    return float(np.minimum.reduce([s[i:len(s) - cw + 1 + i] for i in range(cw)]).max())


def category_series(probs7, cat_weights):
    out = {}
    for cat, weights in cat_weights.items():
        score = np.zeros(probs7.shape[0], dtype=np.float64)
        for lbl, w in weights.items():
            score = np.maximum(score, w * probs7[:, USED.index(lbl)])
        out[cat] = score
    return out


def eval_clips(cat_series, clip_labels, thresholds, cw):
    correct = 0
    fp = miss = 0
    per = {cat: [0, 0, 0] for cat in CATS}
    for i, lab in enumerate(clip_labels):
        fired = {c: clip_score(cat_series[i][c], cw[c]) >= thresholds[c] for c in CATS}
        if lab == "NORMAL":
            bad = any(fired.values())
            correct += 0 if bad else 1
            fp += 1 if bad else 0
        elif fired.get(lab, False):
            correct += 1
        else:
            miss += 1
        for c in CATS:
            if lab == c:
                per[c][0 if fired[c] else 2] += 1
            elif fired[c]:
                per[c][1] += 1
    f1s = []
    for cat, (t, f, n) in per.items():
        p = t / (t + f) if t + f else 0.0
        r = t / (t + n) if t + n else 0.0
        f1 = 2 * p * r / (p + r) if p + r else 0.0
        per[cat] = {"tp": int(t), "fp": int(f), "fn": int(n),
                    "prec": p, "rec": r, "f1": f1}
        f1s.append(f1)
    return {"accuracy": correct / len(clip_labels), "macroF1": float(np.mean(f1s)),
            "falseAlarmClips": fp, "missedClips": miss, "per": per}


def _grid():
    return np.unique(np.concatenate([np.arange(0.001, 1.0, 0.005),
                                     [0.997, 0.999, 0.9995]]))


def tune_thresholds_f1(cat_series, clip_labels, min_precision, cw):
    grid = _grid()
    chosen = {}
    for cat in CATS:
        peaks = np.array([clip_score(cs[cat], cw[cat]) for cs in cat_series])
        y = np.array([1 if l == cat else 0 for l in clip_labels])
        fired = peaks[:, None] >= grid[None, :]
        tp = (fired & (y[:, None] == 1)).sum(0)
        fp = (fired & (y[:, None] == 0)).sum(0)
        fn = (~fired & (y[:, None] == 1)).sum(0)
        prec = tp / np.maximum(tp + fp, 1)
        rec = tp / np.maximum(tp + fn, 1)
        f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
        meets = np.where(prec >= min_precision)[0]
        cand = meets if len(meets) else np.where(grid >= 0.5)[0]
        if not len(cand):
            cand = np.arange(len(grid))
        near = cand[f1[cand] >= f1[cand].max() - 0.01]
        chosen[cat] = float(grid[near[np.lexsort((near, prec[near]))[-1]]])
    return chosen


def tune_thresholds_accuracy(cat_series, clip_labels, min_precision, cw):
    grid = _grid()
    y = np.array(clip_labels)
    is_norm = y == "NORMAL"
    fire = {cat: np.array([clip_score(cs[cat], cw[cat]) for cs in cat_series])[:, None] >= grid[None, :]
            for cat in CATS}
    seed = tune_thresholds_f1(cat_series, clip_labels, min_precision, cw)
    idx = {cat: int(np.argmin(np.abs(grid - seed[cat]))) for cat in CATS}

    def correct_count():
        anyf = np.zeros(len(y), bool)
        for cat in CATS:
            anyf |= fire[cat][:, idx[cat]]
        c = int((is_norm & ~anyf).sum())
        for cat in CATS:
            c += int(((y == cat) & fire[cat][:, idx[cat]]).sum())
        return c

    for _ in range(4):
        for cat in CATS:
            best_i, best_c = idx[cat], correct_count()
            for i in range(len(grid)):
                idx[cat] = i
                c = correct_count()
                if c > best_c:
                    best_c, best_i = c, i
            idx[cat] = best_i
    return {cat: float(grid[idx[cat]]) for cat in CATS}


def fit(spec, Xtr, Ytr):
    if spec["name"] == "linear":
        models = []
        for j in range(len(USED)):
            m = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
            m.fit(Xtr, Ytr[:, j])
            models.append(m)

        def predict(X):
            P = np.zeros((X.shape[0], len(USED)))
            for j, m in enumerate(models):
                P[:, j] = m.predict_proba(X)[:, 1]
            return P
        return predict, models
    mlps = []
    for s in spec["seeds"]:
        m = MLPClassifier(hidden_layer_sizes=spec["hidden"], alpha=1e-4, max_iter=800,
                          early_stopping=True, n_iter_no_change=20, random_state=s)
        m.fit(Xtr, Ytr)
        mlps.append(m)

    def predict(X):
        P = np.zeros((X.shape[0], len(USED)))
        for m in mlps:
            P += m.predict_proba(X)
        return P / len(mlps)
    return predict, mlps


def window_auroc(Y, P):
    aucs = [roc_auc_score(Y[:, j], P[:, j])
            for j in range(len(USED)) if 0 < Y[:, j].sum() < len(Y)]
    return float(np.mean(aucs)) if aucs else float("nan")


def spec_name(spec):
    if spec["name"] == "linear":
        return "linear"
    return f"mlp{spec['hidden'][0]}x{spec['hidden'][1]}_n{len(spec['seeds'])}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=".head_train/features.npz", type=Path)
    ap.add_argument("--config", default="src/main/resources/config/categories.json", type=Path)
    ap.add_argument("--out", default=".head_train/head.npz", type=Path)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--min-precision", type=float, default=0.5)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    d = np.load(root / args.features, allow_pickle=True)
    X = d["X"].astype(np.float64)
    clip_start = d["clip_start"]
    clip_count = d["clip_count"]
    labels = [str(x) for x in d["labels"]]
    classes = [str(x) for x in d["classes"]]
    used_idx = [classes.index(u) for u in USED]
    nclip = len(labels)

    Y = np.zeros((X.shape[0], len(USED)))
    rows_of_clip = []
    for i, lab in enumerate(labels):
        s, c = int(clip_start[i]), int(clip_count[i])
        rows_of_clip.append(np.arange(s, s + c))
        for lbl in CAT_TO_LABELS.get(lab, []):
            Y[s:s + c, USED.index(lbl)] = 1.0

    cfg = json.loads((root / args.config).read_text(encoding="utf-8"))
    cat_weights = {c["category"]: c["labels"] for c in cfg["categories"]}

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    ids = np.arange(nclip)
    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)

    def series_from(P):
        out, pos = [], 0
        for i in ids:
            c = int(clip_count[i])
            out.append(category_series(P[pos:pos + c], cat_weights))
            pos += c
        return out

    # OOF predictions per model (each clip scored by a model that never saw it).
    oof = {}
    for spec in MODELS:
        name = spec_name(spec)
        Poof = np.zeros((X.shape[0], len(USED)))
        for tr, te in skf.split(ids, labels):
            wtr = np.concatenate([rows_of_clip[i] for i in tr])
            wte = np.concatenate([rows_of_clip[i] for i in te])
            predict, _ = fit(spec, Xs[wtr], Y[wtr])
            Poof[wte] = predict(Xs[wte])
        oof[name] = Poof
        print(f"scored OOF for {name}", flush=True)

    results = {}
    for name, Poof in oof.items():
        series = series_from(Poof)
        au = window_auroc(Y, Poof)
        for cwname, cw in CW_CONFIGS.items():
            thr = tune_thresholds_accuracy(series, labels, args.min_precision, cw)
            m = eval_clips(series, labels, thr, cw)
            results[(name, cwname)] = {"thr": thr, "oof": m, "cw": cw,
                                       "auroc": au}
            print(f"[{name:18s} {cwname:11s}] OOF acc={m['accuracy']:.3f} "
                  f"mF1={m['macroF1']:.3f} FPclips={m['falseAlarmClips']} "
                  f"miss={m['missedClips']} AUROC={au:.3f}", flush=True)

    best = max(results, key=lambda k: (results[k]["oof"]["accuracy"],
                                       results[k]["oof"]["macroF1"]))
    name, cwname = best
    print(f"\n== best: model={name} cw={cwname} acc={results[best]['oof']['accuracy']:.3f} "
          f"mF1={results[best]['oof']['macroF1']:.3f} ==", flush=True)
    print("  thr:", {k: round(v, 3) for k, v in results[best]["thr"].items()}, flush=True)
    for cat, mm in results[best]["oof"]["per"].items():
        print(f"    {cat:16s} P={mm['prec']:.2f} R={mm['rec']:.2f} F1={mm['f1']:.2f} "
              f"tp={mm['tp']} fp={mm['fp']} fn={mm['fn']}", flush=True)

    spec = next(s for s in MODELS if spec_name(s) == name)
    cw = results[best]["cw"]
    wall = np.concatenate(rows_of_clip)
    predict, final_model = fit(spec, Xs[wall], Y[wall])
    series_ins = series_from(predict(Xs[wall]))
    thr_final = tune_thresholds_accuracy(series_ins, labels, args.min_precision, cw)
    m_ins = eval_clips(series_ins, labels, thr_final, cw)
    print(f"\nfinal all-data in-sample acc={m_ins['accuracy']:.3f} mF1={m_ins['macroF1']:.3f}",
          flush=True)
    print("  shipped thr:", {k: round(v, 3) for k, v in thr_final.items()}, flush=True)

    save = {"kind": spec["name"], "classes": np.asarray(classes),
            "used": np.asarray(USED), "used_idx": np.asarray(used_idx),
            "mean": scaler.mean_.astype(np.float32),
            "scale": scaler.scale_.astype(np.float32),
            "confirm_windows": np.asarray([cw[c] for c in CATS]),
            "confirm_cats": np.asarray(CATS)}
    if spec["name"] == "linear":
        save["W"] = np.stack([m.coef_[0] for m in final_model]).astype(np.float32)
        save["b"] = np.array([m.intercept_[0] for m in final_model]).astype(np.float32)
    else:
        save["nmodels"] = np.asarray(len(final_model))
        for mi, m in enumerate(final_model):
            save[f"nlayers{mi}"] = np.asarray(len(m.coefs_))
            for li, (w, bb) in enumerate(zip(m.coefs_, m.intercepts_)):
                save[f"W{mi}_{li}"] = w.astype(np.float32)
                save[f"b{mi}_{li}"] = bb.astype(np.float32)
    np.savez(root / args.out, **save)
    (root / args.out).with_suffix(".json").write_text(json.dumps({
        "best": {"model": name, "cw": cwname, "confirm_windows": cw,
                 "oof_accuracy": results[best]["oof"]["accuracy"],
                 "oof_macroF1": results[best]["oof"]["macroF1"],
                 "shipped_thresholds": thr_final, "oof_thresholds": results[best]["thr"],
                 "per_category": results[best]["oof"]["per"]},
        "grid": {f"{k[0]}|{k[1]}": {"acc": v["oof"]["accuracy"],
                                    "mF1": v["oof"]["macroF1"]} for k, v in results.items()},
    }, indent=2), encoding="utf-8")
    print(f"wrote {root / args.out} (model={name}, cw={cwname})", flush=True)


if __name__ == "__main__":
    main()
