# OmniEar

Real-time acoustic threat detection for the desktop.

OmniEar listens to a microphone (or replays a WAV), classifies short audio
windows with an ONNX **CED-mini** audio model, maps the model's 17 labels onto
**6 application threat categories**, and applies explainable temporal decision
logic before raising a prioritised alert. It is a fully offline **Java 21 +
Swing** desktop application for Windows, macOS and Linux. Audio is processed in
memory and never written to disk.

- **Live dashboard** - device picker, level meter, per-category score bars with
  threshold markers, a five-second input diagnostic, and a prioritised alert
  history with CSV export.
- **Six categories** - scream/distress, gunshot, explosion, impact/crash, siren,
  glass break.
- **Explainable decision layer** - per-category thresholds, optional moving-average
  score smoothing, and N-of-M window confirmation with a cooldown.
- **Glass-break detector** - a dedicated high-frequency DSP heuristic that runs
  alongside the model.
- **Command line** - WAV replay, microphone capture, WAV recording and an
  evaluation harness.

## Download and run

Grab the latest `omniear.jar` from the
[**Releases**](https://github.com/johanisdone/omniearjava/releases/latest) page,
then run:

```bash
java -jar omniear.jar --gui
```

Requires **Java 21 or newer** (`java -version`). The jar bundles the ONNX model,
so no extra files are needed. On first launch the app opens your default
microphone; use the device picker to choose another input.

Project site: <https://johanisdone.github.io/omniearjava/> (served from `docs/`
via GitHub Pages).

## Categories and alert priorities

| Category        | Priority | Notes |
|-----------------|----------|-------|
| GUNSHOT         | P0       | Critical |
| EXPLOSION       | P0       | Critical |
| SCREAM_DISTRESS | P1       | High |
| IMPACT_CRASH    | P1       | High |
| GLASS_BREAK     | P1       | High (model + DSP heuristic) |
| SIREN           | P4       | Informational |

## Requirements

- Java 21+ to run the packaged jar.
- JDK 21 + Maven (or the bundled wrapper) to build from source.

## Build from source

```bash
git clone https://github.com/johanisdone/omniearjava.git
cd omniearjava
./mvnw -DskipTests package          # -> target/omniear.jar
java -jar target/omniear.jar --gui
```

On Windows use `mvnw.cmd` instead of `./mvnw`.

Run the test suite (27 tests, including an ONNX parity check):

```bash
./mvnw test
```

## Command line

```
--gui [device]                 open the live dashboard window
--list-devices                 list audio capture devices
--file <wav>                   replay a WAV through the full live pipeline
--mic [device]                 capture from a microphone (console alerts)
--record <wav> [device]        capture microphone audio to a WAV
--seconds <n>                  recording length for --record (default 6)
--rate <hz>                    capture sample-rate override
--eval <dir>                   run the evaluation harness (dir needs labels.csv)
--eval-out <path>              where to write the JSON evaluation report
--models <dir>                 model directory (default: models)
```

`[device]` is a case-insensitive substring matched against the capture-device
names; omit it to use the system default input.

## How it works

1. **Capture** - mono audio is resampled to 16 kHz with a phase-correct streaming
   resampler and sliced into 2.0 s windows with a 1.0 s hop (hop-aligned so no
   window is ever dropped).
2. **Preprocess** - each window is peak-normalised before inference.
3. **Backbone** - `models/ced-mini.onnx` maps a `waveform[*, 32000]` window to
   527 class logits.
4. **Head** - `models/onnx/head.onnx` maps the logits to 17 probabilities.
5. **Map** - `ScoreMapper` reduces the 17 labels to 6 category scores
   (weighted max).
6. **Glass DSP** - `GlassBreakDetector` may raise the glass score to 1.0.
7. **Decide** - `TemporalDecisionEngine` applies the optional score smoothing,
   per-category thresholds, N-of-M confirmation and cooldown, then emits a
   prioritised `ThreatEvent`.

## Model

The runtime uses only the ONNX pair:

- `models/ced-mini.onnx` - CED-mini backbone (logits `[*, 527]`).
- `models/onnx/head.onnx` - 17-class head (`probabilities[*, 17]`).
- `models/onnx/audiodet_v4.json` - upstream configuration.

The GGUF heads under `models/heads/` are **not** numerically compatible with the
ONNX backbone and must not be mixed - see `docs/model-contract.md`.

## Evaluation

Run against a labelled dataset (a directory containing `labels.csv` and an
`audio/` folder):

```bash
./mvnw -DskipTests package
java -jar target/omniear.jar --eval <dataset dir> --eval-out report.json
```

Reported on the bundled real-audio evaluation set:

| Metric        | Honest out-of-fold | In-sample harness |
|---------------|--------------------|-------------------|
| Clip accuracy | 0.889              | 0.990             |
| Macro F1      | 0.810              | 0.992             |

The out-of-fold figures come from `scripts/oof_decision_eval.py` (5-fold,
clip-disjoint) and are the honest, non-leaked numbers; the harness figures score
the shipped head on the same data and are optimistic.

## Project layout

```
src/main/java/com/example/omniear/
  app/         CLI entry point and wiring
  audio/       capture, resampling, rolling window buffer
  detection/   score mapping, temporal decision engine, glass detector
  model/       ONNX backbone + head, preprocessing
  ui/          Swing dashboard
  eval/        evaluation harness
models/        ONNX backbone + head and upstream config
config/        categories.json (thresholds, smoothing, confirmation)
scripts/       feature extraction, head training, decision tuning
docs/          model contract, licensing notes, GitHub Pages site
```

## Releasing

The `Build and release` workflow builds `omniear.jar` on every push and attaches
it to a GitHub Release when a `v*` tag is pushed:

```bash
git tag v0.1.0
git push origin v0.1.0
```

To publish the landing page, enable **Settings > Pages > Source: Deploy from a
branch > master > /docs**.

## Licensing

See `docs/UPSTREAM-LICENSING.md` and `models/LICENSING.md` for upstream model and
dataset attribution.
