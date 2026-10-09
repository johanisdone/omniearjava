"""Temporary Python reference pipeline for AudioDet CED-mini (ONNX path).

NOT part of deployment. Used only to produce known-good scores so the Java
implementation can be validated against them (Phase 1/2 of the blueprint).

IMPORTANT: This uses the ONNX-consistent pairing:
    ced-mini.onnx  (mispeech/ced-mini)  : waveform[*,32000] -> logits[*,527]
    onnx/head.onnx (ShantyCam)          : logits[*,527]    -> probabilities[*,17]
plus onnx/audiodet_v4.json for class order + thresholds.

Do NOT mix with heads/*/head_weights.bin, which are trained on ced.cpp (GGUF)
logits and are numerically inconsistent with the ONNX frontend.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import soundfile as sf

SR = 16000
WIN_SAMPLES = 2 * SR  # 32000
HOP_SAMPLES = 1 * SR  # 16000


class AudioDetReference:
    def __init__(self, models_dir: str | Path):
        models = Path(models_dir)
        self.backbone = ort.InferenceSession(
            str(models / "ced-mini.onnx"), providers=["CPUExecutionProvider"]
        )
        self.head = ort.InferenceSession(
            str(models / "onnx" / "head.onnx"), providers=["CPUExecutionProvider"]
        )
        meta = json.loads((models / "onnx" / "audiodet_v4.json").read_text(encoding="utf-8"))
        self.classes: list[str] = list(meta["classes"])
        self.thresholds: dict[str, float] = dict(meta["thresholds"])
        self.sr = int(meta.get("sr", SR))
        self.win_s = float(meta.get("win_s", 2.0))
        self.hop_s = float(meta.get("hop_s", 1.0))

    # -- audio -------------------------------------------------------------
    @staticmethod
    def load_wav(path: str | Path) -> np.ndarray:
        x, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != SR:
            raise ValueError(f"expected {SR} Hz, got {sr}: {path}")
        return np.ascontiguousarray(x, dtype=np.float32)

    def window_offsets(self, n: int) -> list[int]:
        if n < WIN_SAMPLES:
            return [0]
        return list(range(0, n - WIN_SAMPLES + 1, HOP_SAMPLES))

    def window_probabilities(self, x: np.ndarray):
        """Return (offsets, probs[N,17]) for all 2s windows over x."""
        offs = self.window_offsets(len(x))
        if len(x) < WIN_SAMPLES:
            x = np.pad(x, (0, WIN_SAMPLES - len(x)))
        wins = np.stack([x[o : o + WIN_SAMPLES] for o in offs]).astype(np.float32)
        logits = self.backbone.run(None, {"waveform": wins})[0]
        probs = self.head.run(None, {"logits": logits})[0]
        return offs, probs

    def clip_scores(self, probs: np.ndarray) -> np.ndarray:
        """two-consecutive-window hysteresis: max_t min(p[t], p[t+1])."""
        if len(probs) == 1:
            return probs[0].copy()
        return np.minimum(probs[:-1], probs[1:]).max(axis=0)

    def fired(self, clip_scores: np.ndarray) -> list[str]:
        return [
            c
            for c, s in zip(self.classes, clip_scores)
            if float(s) >= float(self.thresholds[c])
        ]

    # -- high level --------------------------------------------------------
    def infer_file(self, path: str | Path) -> dict:
        x = self.load_wav(path)
        offs, probs = self.window_probabilities(x)
        clip = self.clip_scores(probs)
        return {
            "file": Path(path).name,
            "sr": SR,
            "num_samples": int(len(x)),
            "window_offsets": [int(o) for o in offs],
            "classes": self.classes,
            "window_probabilities": [[float(v) for v in row] for row in probs],
            "clip_scores": [float(v) for v in clip],
            "fired": self.fired(clip),
        }


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("wav", help="16 kHz mono WAV file")
    ap.add_argument("--models", default=str(Path(__file__).resolve().parent.parent / "models"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ref = AudioDetReference(args.models)
    result = ref.infer_file(args.wav)
    text = json.dumps(result, indent=2)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
    else:
        print(text)


if __name__ == "__main__":
    main()
