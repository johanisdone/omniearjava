"""Per-category threshold sweep over the labelled eval set.

Replicates the app's decision pipeline exactly:
  - ScoreMapper       score(category) = max over mapped labels of weight * p[label]
  - TemporalDecision  a category fires on a clip iff >= confirmWindows consecutive
                      windows score >= threshold; cooldown only gates repeat events.
  - Harness metrics   clip-level tp/fp/fn per category, macroF1, and total alert
                      events / hour on NORMAL clips.

Each category's threshold is chosen by argmax F1, but only among thresholds that
meet --min-precision (an alerting system must not cry wolf). If precision is
unreachable for a category, a conservative non-trivial threshold is used instead
(fallback-min-threshold) rather than "always fire".

Per-window category scores are computed once per clip with the ONNX reference
pipeline and reused in-memory for the sweep. For each category the argmax-F1
threshold is chosen (ties resolved toward the higher threshold), then the chosen
set is replayed with full event simulation to report macroF1 and false
alerts/hour, and a ready-to-paste categories.json fragment is printed.

Run with the reference interpreter (has onnxruntime + soundfile):
  python scripts/threshold_sweep.py
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

SR = 16000
WIN = 32000
HOP = 16000


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


class Model:
    def __init__(self, models_dir):
        m = Path(models_dir)
        self.backbone = ort.InferenceSession(str(m / "ced-mini.onnx"),
                                             providers=["CPUExecutionProvider"])
        self.head = ort.InferenceSession(str(m / "onnx" / "head.onnx"),
                                         providers=["CPUExecutionProvider"])
        meta = json.loads((m / "onnx" / "audiodet_v4.json").read_text(encoding="utf-8"))
        self.classes = list(meta["classes"])

    def probs(self, x):
        offs = window_offsets(len(x))
        if len(x) < WIN:
            x = np.pad(x, (0, WIN - len(x)))
        wins = np.stack([x[o:o + WIN] for o in offs]).astype(np.float32)
        logits = self.backbone.run(None, {"waveform": wins})[0]
        return self.head.run(None, {"logits": logits})[0]


def simulate_events(scores, times, thresh, confirm, cooldown):
    events = []
    count = 0
    first = -1
    peak = 0.0
    last_fire = None
    for s, t in zip(scores, times):
        if s >= thresh:
            if count == 0:
                first = t
            count += 1
            peak = max(peak, s)
            if count >= confirm and (last_fire is None or (t - last_fire) >= cooldown):
                events.append((first, t, peak))
                last_fire = t
                count = 0
                first = -1
                peak = 0.0
        else:
            count = 0
            first = -1
            peak = 0.0
    return events


def f1p(p, r):
    return 2 * p * r / (p + r) if p + r else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="eval-real", type=Path)
    ap.add_argument("--models", default="models", type=Path)
    ap.add_argument("--config", default="src/main/resources/config/categories.json", type=Path)
    ap.add_argument("--start", type=float, default=0.01)
    ap.add_argument("--stop", type=float, default=0.995)
    ap.add_argument("--step", type=float, default=0.005)
    ap.add_argument("--show", type=int, default=3, help="candidates printed per category")
    ap.add_argument("--min-precision", type=float, default=0.5,
                    help="only consider thresholds at/above this precision")
    ap.add_argument("--fallback-min-threshold", type=float, default=0.5,
                    help="if precision floor is unreachable, stay at/above this")
    ap.add_argument("--f1-tolerance", type=float, default=0.01,
                    help="among thresholds within this F1 of the max, prefer precision")
    ap.add_argument("--cache", default=Path(".sweep_cache.pkl"), type=Path,
                    help="score cache inside --eval dir (gitignored)")
    ap.add_argument("--refresh", action="store_true", help="rescore instead of using cache")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    cfg = json.loads((root / args.config).read_text(encoding="utf-8"))
    cats = cfg["categories"]
    labels = [c["category"] for c in cats]

    model = Model(root / args.models)
    label_index = {cls: i for i, cls in enumerate(model.classes)}

    rows = []
    for line in (root / args.eval / "labels.csv").read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        rel, lab = line.split(",")[:2]
        rows.append((root / args.eval / rel, lab))

    clipdata = []
    cache_path = root / args.eval / args.cache
    if cache_path.exists() and not args.refresh:
        import pickle
        clipdata = pickle.loads(cache_path.read_bytes())
        if len(clipdata) != len(rows):
            print(f"  cache size {len(clipdata)} != rows {len(rows)}; rescoring", flush=True)
            clipdata = []
        else:
            print(f"  loaded {len(clipdata)} cached clip scores", flush=True)
    if not clipdata:
        print("  scoring clips ...", flush=True)
        for idx, (path, lab) in enumerate(rows):
            if not path.is_file():
                sys.exit(f"missing {path}")
            x = load_audio(path)
            probs = model.probs(x)
            offs = window_offsets(len(x))
            times = [((o + WIN) * 1000) // SR for o in offs]
            series = {}
            for cat in cats:
                rule_scores = np.zeros(len(probs), dtype=np.float64)
                for lbl, w in cat["labels"].items():
                    rule_scores = np.maximum(rule_scores, w * probs[:, label_index[lbl]])
                series[cat["category"]] = rule_scores
            clipdata.append((lab, len(x) / SR, times, series))
            if (idx + 1) % 200 == 0:
                print(f"  scored {idx + 1}/{len(rows)} clips", flush=True)
        import pickle
        cache_path.write_bytes(pickle.dumps(clipdata))

    grid = np.unique(np.concatenate(
        [np.arange(args.start, args.stop + 1e-9, args.step), [0.997, 0.999, 0.9995]]))
    neg_hours = sum(d for lab, d, _, _ in clipdata if lab == "NORMAL") / 3600.0

    chosen = {}
    print(f"\n=== sweep: best {args.show} candidates per category (F1 ordering) ===", flush=True)
    for cat in cats:
        peaks = np.array([np.minimum(c[3][cat["category"]][:-1], c[3][cat["category"]][1:]).max().item()
                          if len(c[3][cat["category"]]) >= 2 else -1.0 for c in clipdata])
        y = np.array([1 if c[0] == cat["category"] else 0 for c in clipdata])
        fired = peaks[:, None] >= grid[None, :]
        tp = (fired & (y[:, None] == 1)).sum(axis=0)
        fp = (fired & (y[:, None] == 0)).sum(axis=0)
        fn = (~fired & (y[:, None] == 1)).sum(axis=0)
        prec = np.divide(tp, np.maximum(tp + fp, 1))
        rec = np.divide(tp, np.maximum(tp + fn, 1))
        f1 = np.divide(2 * prec * rec, np.maximum(prec + rec, 1e-9))
        meets = np.where(prec >= args.min_precision)[0]
        if len(meets):
            cand = meets
            note = ""
        else:
            pool = np.where(grid >= args.fallback_min_threshold)[0]
            cand = pool if len(pool) else np.arange(len(grid))
            note = (f"  [warn: P>={args.min_precision} unreachable for any threshold; "
                    f"chose conservative t>={args.fallback_min_threshold}]")
        near = cand[f1[cand] >= f1[cand].max() - args.f1_tolerance]
        best = near[np.lexsort((near, prec[near]))[-1]]
        chosen[cat["category"]] = float(grid[best])
        print(f"\n{cat['category']:16s} chosen t={grid[best]:.3f} "
              f"P={prec[best]:.2f} R={rec[best]:.2f} F1={f1[best]:.3f} "
              f"tp={int(tp[best])} fp={int(fp[best])} fn={int(fn[best])}{note}")
        shown = set()
        order = np.lexsort((grid, f1))
        for i in range(len(grid) - 1, -1, -1):
            key = (round(prec[order[i]], 2), round(rec[order[i]], 2), round(f1[order[i]], 3))
            if key in shown or len(shown) >= args.show:
                continue
            shown.add(key)
            print(f"  t={grid[order[i]]:0.3f}  P={prec[order[i]]:.2f} R={rec[order[i]]:.2f} "
                  f"F1={f1[order[i]]:.3f} fp={int(fp[order[i]])}")

    print("\n=== chosen set: full event simulation ===", flush=True)
    f1s = []
    false_events = 0
    for cat in cats:
        name = cat["category"]
        t = chosen[name]
        confirm = cat["confirmWindows"]
        cooldown = cat["cooldownMs"]
        tp = fp = fn = 0
        for lab, dur, times, series in clipdata:
            events = simulate_events(series[name], times, t, confirm, cooldown)
            fired = bool(events)
            if lab == name:
                if fired:
                    tp += 1
                else:
                    fn += 1
            elif fired:
                fp += 1
                if lab == "NORMAL":
                    false_events += len(events)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = f1p(prec, rec)
        f1s.append(f1)
        print(f"  {name:16s} t={t:.3f} P={prec:.2f} R={rec:.2f} F1={f1:.3f} "
              f"tp={tp} fp={fp} fn={fn}")
    counted = [f for f in f1s if f > 0]
    macro = sum(f1s) / len(f1s) if f1s else 0.0
    fp_hr = false_events / neg_hours if neg_hours else 0.0
    print(f"macroF1={macro:.4f}  false alerts/hr (NORMAL, {neg_hours:.2f} h)="
          f"{fp_hr:.2f}  total={false_events}")

    print("\n=== recommended categories.json thresholds ===", flush=True)
    out = {"uncertainFloor": cfg.get("uncertainFloor", 0.3), "categories": []}
    for cat in cats:
        rec = {"category": cat["category"], "labels": cat["labels"],
               "threshold": round(chosen[cat["category"]], 3),
               "confirmWindows": cat["confirmWindows"], "cooldownMs": cat["cooldownMs"]}
        out["categories"].append(rec)
        print(json.dumps(rec))
    sys_path = root / "eval-real" / "categories.recommended.json"
    sys_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {sys_path}")


if __name__ == "__main__":
    main()