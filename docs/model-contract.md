# AudioDet CED-mini — Model Contract (verified)

Verified against the live Hugging Face repos during implementation. Re-check at
deploy time; artifacts and licenses can change.

## Selected artifacts (ONNX path)

| Role | File | Source |
|---|---|---|
| Backbone | `models/ced-mini.onnx` | `mispeech/ced-mini` → `model.onnx` |
| Head | `models/onnx/head.onnx` (+ `head.onnx.data`) | `ShantyCam/audiodet-ced-mini` |
| Metadata | `models/onnx/audiodet_v4.json` | `ShantyCam/audiodet-ced-mini` |

License: backbone and ShantyCam heads/ONNX/recipe are **Apache-2.0** (see
`docs/UPSTREAM-LICENSING.md`, `docs/UPSTREAM-NOTICE.txt`).

## Tensor interface

```
ced-mini.onnx  (producer: PyTorch 2.10, opset 19, IR 10)
    IN  waveform  [batch_size, audio_samples]  float32   (raw PCM, NOT log-mel)
    OUT logits    [batch_size, 527]            float32
    graph contains STFT  -> the mel frontend is internal

onnx/head.onnx
    IN  logits       [b, 527]  float32
    OUT probabilities[b, 17]   float32
    StandardScaler -> Linear(512)->ReLU -> Linear(256)->ReLU -> Linear(128)->ReLU -> Linear(17) -> Sigmoid
```

## Preprocessing (must match exactly)

- Sample rate: **16000 Hz**
- Channels: **mono** (average channels)
- PCM scaling: signed 16-bit / 32768 → float in [-1, 1]
- Window: **2.0 s (32000 samples)**, Hop: **1.0 s (16000 samples)**
- Padding: if a signal is shorter than one window, zero-pad to 32000 (single window)
- **No external log-mel**: the backbone computes the spectrogram internally
- Feeding raw PCM to the backbone is correct (input is literally named `waveform`)

## Class order (17)

```
siren, horn, gunshot, glass, knock, loud_voice, dog_bark, baby_cry, chainsaw,
drilling, smoke_alarm, speech, cat, explosion, vehicle_crash, scream, music
```

## Decision rule

Per clip/window-sequence: `clip_score[c] = max_t min(p[t][c], p[t+1][c])`
(two-consecutive-window hysteresis). A class fires when `clip_score >= threshold`
from `audiodet_v4.json`.

## CRITICAL: two incompatible heads — do not mix

The recipe (`recipe/train_v4_ced.py`) reports a parity test where the **mispeech
ONNX frontend** and **ced.cpp (GGUF)** produce different 527-dim logits for the
same 2 s window (Δprob up to 0.5). Consequently there are two head families:

- **ONNX head** `onnx/head.onnx` + `onnx/audiodet_v4.json` — used here.
- **GGUF heads** `heads/v4_ced/head_weights.bin`, `heads/v4_distill/head_weights.bin`
  (+ `head_meta.json`) — trained on **ced.cpp q8 logits**. These are the
  recommended production heads for a ced.cpp/GGUF runtime and are **not
  numerically consistent with the ONNX backbone**.

Throughout this project we use only the ONNX pair above. `head_weights.bin` is
retained solely as a reference/cross-check artifact.

## Runtime version pinning

ONNX numeric results depend on the runtime build. Python reference used
`onnxruntime==1.29.0` to match Java `com.microsoft.onnxruntime:onnxruntime:1.29.0`.
Observed max probability deviation before pinning (Python 1.31 vs Java 1.29) was
~0.024 on a transient-heavy fixture; after pinning, max deviation is < 1e-3.

## Semantic validation (smoke test)

ONNX pipeline on `ShantyCam/audiodet-synth-smoke` clips (8 s, 16 kHz):
`smoke_alarm` = 1.0, all other classes ≈ 0. Digital silence is
out-of-distribution and can yield spurious non-zero scores — validate with real
audio, not synthetic silence.
