"""Honest (out-of-fold) decision-layer evaluation for the shipped head.

The harness / decision_lab score the eval set with the head that was fit on that
same data, so its probabilities are saturated (in-sample). This script instead
reproduces the OOF predictions used by train_head.py (5-fold, clip-disjoint) and
runs the SAME decision logic (incl. confirmOf) over them, so tuning/validation is
not leaked through the head.

Reuses functions from decision_lab (import-safe, no onnxruntime needed here).

Run with the SYSTEM interpreter (needs scikit-learn):
    python scripts/oof_decision_eval.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import decision_lab as dl  # noqa: E402

USED = ["scream", "gunshot", "explosion", "vehicle_crash", "knock", "siren", "glass"]


def search_keep_cof(series_list, labels, cat, grid, candidates, cw, cof):
    """Search transform + threshold only, keeping a fixed confirmWindows/confirmOf."""
    y = np.array([1 if l == cat else 0 for l in labels])
    best = None
    for kind, param in (candidates if candidates is not None else dl.ALL_CANDIDATES):
        f1, t, p = dl.best_threshold_for(series_list, y, kind, param, cw, cof, grid)
        key = (round(f1, 4), round(p, 4))
        cand = {"key": key, "f1": f1, "threshold": float(t), "transform": kind, "param": param,
                "confirmWindows": cw, "confirmOf": cof, "precision": p}
        if best is None or key > best["key"]:
            best = cand
    return best


def predict_proba7(model, X):
    pr = model.predict_proba(X)
    if isinstance(pr, list):
        return np.stack([p[:, 1] for p in pr], axis=1)
    if pr.ndim == 2 and pr.shape[1] == len(USED):
        return pr
    raise RuntimeError(f"unexpected predict_proba shape: {np.shape(pr)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--transforms", default="", help="comma list to restrict, e.g. 'none' or 'none,movavg'")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--test-frac", type=float, default=0.4)
    ap.add_argument("--config", default="", help="evaluate this categories.json instead of searching")
    ap.add_argument("--keep-cof", action="store_true",
                    help="keep shipped confirmWindows/confirmOf (2-of-3) per category; "
                         "search transform + threshold only")
    args = ap.parse_args()
    candidates = None
    if args.transforms:
        allowed = set(args.transforms.split(","))
        candidates = [(k, p) for (k, p) in dl.ALL_CANDIDATES if k in allowed]

    root = Path(__file__).resolve().parent.parent
    d = np.load(root / ".head_train" / "features.npz", allow_pickle=True)
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
        for cat in dl.CATS:
            pass
    # label->USED rows
    cat_to_used = {
        "SCREAM_DISTRESS": ["scream"], "GUNSHOT": ["gunshot"], "EXPLOSION": ["explosion"],
        "IMPACT_CRASH": ["vehicle_crash", "knock"], "SIREN": ["siren"], "GLASS_BREAK": ["glass"],
    }
    for i, lab in enumerate(labels):
        s, c = int(clip_start[i]), int(clip_count[i])
        for lbl in cat_to_used.get(lab, []):
            Y[s:s + c, USED.index(lbl)] = 1.0

    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)
    ids = np.arange(nclip)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    Poof = np.zeros((X.shape[0], len(USED)))
    for tr, te in skf.split(ids, labels):
        wtr = np.concatenate([rows_of_clip[i] for i in tr])
        wte = np.concatenate([rows_of_clip[i] for i in te])
        m = MLPClassifier(hidden_layer_sizes=(256, 128), alpha=1e-4, max_iter=800,
                          early_stopping=True, n_iter_no_change=20, random_state=0)
        m.fit(Xs[wtr], Y[wtr])
        Poof[wte] = predict_proba7(m, Xs[wte])
    print("OOF scores ready", flush=True)

    shipped = json.loads((root / "src/main/resources/config/categories.json").read_text(encoding="utf-8"))
    cfg = json.loads((root / args.config).read_text(encoding="utf-8")) if args.config else shipped
    cats = cfg["categories"]
    weights = {c["category"]: c["labels"] for c in cats}

    series = []
    for i in range(nclip):
        s, c = int(clip_start[i]), int(clip_count[i])
        probs7 = Poof[s:s + c]
        ser = {}
        for cat, wl in weights.items():
            v = np.zeros(c)
            for lbl, w in wl.items():
                v = np.maximum(v, w * probs7[:, USED.index(lbl)])
            ser[cat] = v
        series.append(ser)

    cal, test = dl.stratified_split(labels, args.test_frac, args.seed)
    print(f"clips={nclip} calibration={len(cal)} test={len(test)}")

    def restrict(idxs):
        return [series[i] for i in idxs], [labels[i] for i in idxs]

    cal_s, cal_l = restrict(cal)
    test_s, test_l = restrict(test)

    base_all = dl.evaluate(series, labels, shipped)
    base_test = dl.evaluate(test_s, test_l, shipped)
    print("\n=== baseline (shipped thresholds) on OOF ===")
    print(f"all  acc={base_all['accuracy']:.3f} macroF1={base_all['macroF1']:.3f} "
          f"FP={base_all['falseAlarmClips']} missed={base_all['missed']}")
    for cat in dl.CATS:
        b = base_all["per"][cat]
        print(f"  {cat:16s} P={b['prec']:.2f} R={b['rec']:.2f} F1={b['f1']:.2f} tp={b['tp']} fp={b['fp']} fn={b['fn']}")

    if args.config:
        cand_all = dl.evaluate(series, labels, cfg)
        cand_test = dl.evaluate(test_s, test_l, cfg)
        print(f"\n=== candidate {args.config} on OOF ===")
        print(f"all  acc={cand_all['accuracy']:.3f} macroF1={cand_all['macroF1']:.3f} "
              f"FP={cand_all['falseAlarmClips']} missed={cand_all['missed']}")
        for cat in dl.CATS:
            b = base_test["per"][cat]
            t = cand_test["per"][cat]
            print(f"  {cat:16s} base P={b['prec']:.2f} R={b['rec']:.2f} F1={b['f1']:.2f}  ->  "
                  f"cand P={t['prec']:.2f} R={t['rec']:.2f} F1={t['f1']:.2f}")
        print(f"test acc base={base_test['accuracy']:.3f} cand={cand_test['accuracy']:.3f}; "
              f"test macroF1 base={base_test['macroF1']:.3f} cand={cand_test['macroF1']:.3f}")
        return

    grid = np.unique(np.concatenate([np.arange(0.05, 1.0, 0.01), [0.999]]))
    tuned = {"uncertainFloor": cfg.get("uncertainFloor", 0.3), "categories": []}
    print("\n=== per-category search on OOF calibration ===")
    for c in cats:
        cat = c["category"]
        if args.keep_cof:
            cw = c["confirmWindows"]
            cof = c.get("confirmOf", cw)
            best = search_keep_cof([s[cat] for s in cal_s], cal_l, cat, grid,
                                   candidates, cw, cof)
        else:
            best = dl.search_category([s[cat] for s in cal_s], cal_l, cat, grid, candidates)
        tuned["categories"].append({
            "category": cat, "labels": c["labels"], "threshold": round(best["threshold"], 3),
            "transform": best["transform"], "param": best["param"],
            "confirmWindows": best["confirmWindows"], "confirmOf": best["confirmOf"],
            "cooldownMs": c["cooldownMs"]})
        print(f"  {cat:16s} t={best['threshold']:.3f} {best['transform']} cw={best['confirmWindows']} "
              f"cof={best['confirmOf']} calF1={best['f1']:.3f}")

    tuned_all = dl.evaluate(series, labels, tuned)
    tuned_test = dl.evaluate(test_s, test_l, tuned)
    print("\n=== tuned (OOF-tuned) config ===")
    print(f"all  acc={tuned_all['accuracy']:.3f} macroF1={tuned_all['macroF1']:.3f} "
          f"FP={tuned_all['falseAlarmClips']} missed={tuned_all['missed']}")
    for cat in dl.CATS:
        b = base_test["per"][cat]
        t = tuned_test["per"][cat]
        print(f"  {cat:16s} base P={b['prec']:.2f} R={b['rec']:.2f} F1={b['f1']:.2f}  ->  "
              f"tuned P={t['prec']:.2f} R={t['rec']:.2f} F1={t['f1']:.2f}")
    print(f"\ntest acc base={base_test['accuracy']:.3f} tuned={tuned_test['accuracy']:.3f}; "
          f"test macroF1 base={base_test['macroF1']:.3f} tuned={tuned_test['macroF1']:.3f}")

    out = root / ".decision_lab"
    out.mkdir(exist_ok=True)
    (out / "oof_report.json").write_text(json.dumps({
        "baseline_all": base_all, "baseline_test": base_test,
        "tuned_all": tuned_all, "tuned_test": tuned_test, "tuned_config": tuned}, indent=2) + "\n",
        encoding="utf-8")
    (out / "oof_categories.candidate.json").write_text(json.dumps(tuned, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out / 'oof_report.json'}")


if __name__ == "__main__":
    main()
