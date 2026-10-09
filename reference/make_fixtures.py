"""Generate synthetic 16 kHz mono WAV fixtures + golden reference JSON.

Synthetic signals are sufficient to prove Java<->Python numerical parity.
They are NOT meant to validate detection semantics.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import soundfile as sf

from audiodet_reference import SR, AudioDetReference

HERE = Path(__file__).resolve().parent
FIX = HERE / "fixtures"
JAVA_AUDIO = HERE.parent / "src" / "test" / "resources" / "audio"


def silence(seconds=2.0):
    return np.zeros(int(SR * seconds), dtype=np.float32)


def white_noise(seconds=2.0, seed=1, amp=0.05):
    rng = np.random.default_rng(seed)
    return (amp * rng.standard_normal(int(SR * seconds))).astype(np.float32)


def tone_sweep(seconds=3.0, f0=600.0, f1=1200.0):
    t = np.arange(int(SR * seconds)) / SR
    freq = np.linspace(f0, f1, len(t))
    phase = 2 * np.pi * np.cumsum(freq) / SR
    return (0.3 * np.sin(phase)).astype(np.float32)


def impulses(seconds=3.0, n=4, seed=2):
    rng = np.random.default_rng(seed)
    x = np.zeros(int(SR * seconds), dtype=np.float32)
    for _ in range(n):
        i = rng.integers(0, len(x) - 200)
        x[i : i + 64] += rng.uniform(0.5, 1.0, size=64).astype(np.float32)
    return x


def mixed(seconds=4.0, seed=3):
    base = white_noise(seconds, seed=seed, amp=0.02)
    t = np.arange(len(base)) / SR
    burst = np.zeros_like(base)
    m = (t > 1.0) & (t < 1.8)
    burst[m] = (0.4 * np.sin(2 * np.pi * 900 * t[m])).astype(np.float32)
    return (base + burst).astype(np.float32)


FIXTURES = {
    "silence_2s.wav": silence(2.0),
    "noise_2s.wav": white_noise(2.0),
    "tone_sweep_3s.wav": tone_sweep(3.0),
    "impulses_3s.wav": impulses(3.0),
    "mixed_4s.wav": mixed(4.0),
}


def main() -> None:
    FIX.mkdir(parents=True, exist_ok=True)
    JAVA_AUDIO.mkdir(parents=True, exist_ok=True)
    ref = AudioDetReference(HERE.parent / "models")
    for name, sig in FIXTURES.items():
        wav = FIX / name
        sf.write(str(wav), sig, SR, subtype="PCM_16")
        result = ref.infer_file(wav)
        golden = FIX / (Path(name).stem + ".json")
        golden.write_text(json.dumps(result, indent=2), encoding="utf-8")
        shutil.copy2(wav, JAVA_AUDIO / name)
        shutil.copy2(golden, JAVA_AUDIO / golden.name)
        print(f"{name}: windows={len(result['window_offsets'])} fired={result['fired']}")
    print(f"\nfixtures -> {FIX}")
    print(f"java test resources -> {JAVA_AUDIO}")


if __name__ == "__main__":
    main()
