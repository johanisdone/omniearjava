"""Extract ced-mini 527-d logits for every window of the labelled eval set.

The retrained head consumes exactly what the Java pipeline feeds to head.onnx:
the ced-mini backbone logits for each 2 s / 1 s-hop window, computed with the
same preprocessing as AudioPreprocessor / audiodet_reference.

Run with the reference interpreter (onnxruntime + soundfile):
    python scripts/extract_features.py
Output: .head_train/features.npz  (X, clip_start, clip_count, labels, classes)
"""
from __future__ import annotations

import argparse
import json
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default="eval-real", type=Path)
    ap.add_argument("--models", default="models", type=Path)
    ap.add_argument("--out", default=".head_train/features.npz", type=Path)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    out = root / args.out
    if out.exists() and not args.refresh:
        print(f"exists: {out} (use --refresh to rebuild)")
        return

    models = root / args.models
    meta = json.loads((models / "onnx" / "audiodet_v4.json").read_text(encoding="utf-8"))
    classes = list(meta["classes"])
    backbone = ort.InferenceSession(str(models / "ced-mini.onnx"),
                                    providers=["CPUExecutionProvider"])

    rows = []
    for line in (root / args.eval / "labels.csv").read_text(encoding="utf-8").splitlines()[1:]:
        if not line.strip():
            continue
        rel, lab = line.split(",")[:2]
        rows.append((root / args.eval / rel, lab))

    X = []
    clip_start = []
    clip_count = []
    labels = []
    for idx, (path, lab) in enumerate(rows):
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        x = load_audio(path)
        offs = window_offsets(len(x))
        if len(x) < WIN:
            x = np.pad(x, (0, WIN - len(x)))
        wins = np.stack([x[o:o + WIN] for o in offs]).astype(np.float32)
        logits = backbone.run(None, {"waveform": wins})[0].astype(np.float32)
        clip_start.append(len(X))
        clip_count.append(len(logits))
        X.extend(logits)
        labels.append(lab)
        if (idx + 1) % 200 == 0:
            print(f"  {idx + 1}/{len(rows)} clips, {len(X)} windows", flush=True)

    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        X=np.asarray(X, dtype=np.float32),
        clip_start=np.asarray(clip_start, dtype=np.int64),
        clip_count=np.asarray(clip_count, dtype=np.int64),
        labels=np.asarray(labels),
        classes=np.asarray(classes),
    )
    print(f"wrote {out}: X={np.asarray(X).shape} clips={len(labels)} windows={len(X)}")


if __name__ == "__main__":
    main()
