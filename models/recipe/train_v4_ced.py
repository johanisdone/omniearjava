#!/usr/bin/env python3
"""AudioDet v4.2 — hlava trénovaná na PRODUKČNÍ reprezentaci: ced.cpp q8 GGUF.

Zjištění z parity testů: ONNX export (mispeech) a ced.cpp (referenční CED
chování) se na 2s oknech liší (Δprob až 0.5) — dva frontendy, dvě pravdy.
Produkce pojede ced.cpp ⇒ hlava se učí na jeho výstupech (codex bod #10).
Kvantizační šum q8 vs f16 je ≤0.015 ⇒ q8 OK.

Reuse z train_v4: plán cropů (identické offsety/aug seedy přes cache),
train_mlp, prahy. Extrakce: ced-batch (proud f32 oken → 527 prob).
Kalibrace: klipová, per-class negativa (falešné TÉŽE třídy + bg).
Export: head_weights.bin (flat f32) + head_meta.json pro betterC loader.

Výstup: ~/audiodet_review/model_v4_ced/
"""
import glob, json, os, pickle, re, struct, subprocess, sys, threading

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_v2 import train_mlp, predict, pick_thresholds, prf  # noqa: E402
from train_v4 import (REVIEW, SEED, SR, WIN, HOP, AUG_CLASSES,  # noqa: E402
                      POS_CROPS, NEG_CROPS, AUG_CROPS, Augmenter,
                      collect_files, file_split, crops_for, crop, load16)
import random  # noqa: E402

OUT_DIR = os.path.join(REVIEW, "model_v4_ced")
CACHE_V4 = os.path.join(REVIEW, "emb_v4.pkl")        # offsety cropů (reuse!)
CACHE_CED = os.path.join(REVIEW, "emb_v4ced.pkl")
BATCH_BIN = os.path.expanduser("~/audiodet_datasets/ced.cpp/build/examples/batch/ced-batch")
GGUF = os.path.expanduser("~/audiodet_datasets/ced-mini-q8.gguf")
NCLS_CED = 527
TARGET_PRECISION = 0.90
NEG_QUANTILE = 0.995


class CedBatch:
    """Streamovací klient ced-batch: okna f32 → 527 prob (lockstep přes vlákno)."""

    def __init__(self, threads=8):
        self.p = subprocess.Popen([BATCH_BIN, GGUF, "--threads", str(threads)],
                                  stdin=subprocess.PIPE, stdout=subprocess.PIPE)

    def run(self, wins):
        """wins: np.ndarray [N, WIN] f32 → [N, 527] f32."""
        n = len(wins)
        payload = np.ascontiguousarray(wins, dtype=np.float32).tobytes()
        t = threading.Thread(target=self.p.stdin.write, args=(payload,))
        t.start()
        out = self.p.stdout.read(n * NCLS_CED * 4)
        t.join()
        return np.frombuffer(out, dtype=np.float32).reshape(n, NCLS_CED).copy()

    def close(self):
        self.p.stdin.close()
        self.p.wait()


def build_plan(items, tr_files):
    """Identický plán cropů jako train_v4 (offsety z jeho cache, kde jsou)."""
    v4cache = pickle.load(open(CACHE_V4, "rb")) if os.path.exists(CACHE_V4) else {}
    rng = random.Random(SEED + 7)
    plan = []
    for i, (path, cls, _src) in enumerate(items):
        n_pos = POS_CROPS if cls else NEG_CROPS
        key0 = (path, "offs", n_pos)
        if key0 not in v4cache:
            try:
                x = load16(path)
            except Exception:
                continue
            v4cache[key0] = crops_for(x, n_pos // 2, n_pos - n_pos // 2, rng) \
                if cls else crops_for(x, 0, n_pos, rng)
        for off in v4cache[key0]:
            plan.append(((path, off, None), i, off, None))
        if i in tr_files and cls in AUG_CLASSES:
            for a in range(AUG_CROPS):
                off = v4cache[key0][a % len(v4cache[key0])]
                plan.append(((path, off, a), i, off, SEED + i * 13 + a))
    return plan


def extract(plan, items):
    cache = pickle.load(open(CACHE_CED, "rb")) if os.path.exists(CACHE_CED) else {}
    todo = [(k, i, off, aug) for k, i, off, aug in plan if k not in cache]
    print(f"extrakce ced.cpp: {len(plan)} cropů, {len(todo)} nových", flush=True)
    if not todo:
        return cache
    batch = CedBatch()
    augmenter = Augmenter(random.Random(SEED + 9))
    wav_cache = {}
    B = 64
    import time
    t0 = time.time()
    for s in range(0, len(todo), B):
        chunk = todo[s:s + B]
        wins = []
        for k, i, off, aug in chunk:
            path = items[i][0]
            if path not in wav_cache:
                if len(wav_cache) > 384:
                    wav_cache.clear()
                try:
                    wav_cache[path] = load16(path)
                except Exception:
                    wav_cache[path] = np.zeros(WIN, dtype=np.float32)
            c = crop(wav_cache[path], off)
            if aug is not None:
                augmenter.rng.seed(aug)
                c = augmenter(c)
            wins.append(c)
        probs = batch.run(np.stack(wins))
        for (k, _i, _o, _a), row in zip(chunk, probs):
            cache[k] = row
        if (s // B) % 40 == 0 and s:
            el = time.time() - t0
            print(f"  [{s}/{len(todo)}] {el:.0f}s, ETA {el/s*(len(todo)-s):.0f}s", flush=True)
            pickle.dump(cache, open(CACHE_CED, "wb"))
    batch.close()
    pickle.dump(cache, open(CACHE_CED, "wb"))
    return cache


def clip_scores(batch, model, scaler, active, path):
    x = load16(path)
    offs = list(range(0, max(len(x) - WIN, 0) + 1, HOP)) or [0]
    wins = np.stack([crop(x, o) for o in offs])
    probs = predict(model, scaler, batch.run(wins))
    if len(probs) == 1:
        return probs[0]
    return np.minimum(probs[:-1], probs[1:]).max(axis=0)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    items = collect_files()
    tr_files, te_files = file_split(items)
    plan = build_plan(items, tr_files)
    cache = extract(plan, items)

    from train_v4 import V4_TYPES
    active = [t for t in V4_TYPES if any(c == t for _p, c, _s in items)]
    X, Y, is_tr = [], [], []
    for k, i, _off, aug in plan:
        if k not in cache:
            continue
        _p, cls, _s = items[i]
        y = np.zeros(len(active), dtype=np.float32)
        if cls:
            if cls not in active:
                continue
            y[active.index(cls)] = 1.0
        X.append(cache[k]); Y.append(y)
        is_tr.append(i in tr_files or aug is not None)
    X = np.stack(X); Y = np.stack(Y); is_tr = np.array(is_tr)
    tr_idx = np.where(is_tr)[0]; te_idx = np.where(~is_tr)[0]
    print(f"matice: {len(X)} cropů (train {len(tr_idx)}, eval {len(te_idx)}), tříd {len(active)}")

    model, scaler = train_mlp(X[tr_idx], Y[tr_idx], len(active))
    thr0 = pick_thresholds(predict(model, scaler, X[tr_idx]), Y[tr_idx], active)
    res_crop = prf(predict(model, scaler, X[te_idx]), Y[te_idx], thr0, active)
    print("\n=== CROP HELD-OUT (2s, ced.cpp q8) ===")
    for t in active:
        r = res_crop[t]
        if r["tp"] + r["fn"] or r["fp"]:
            p = "—" if r["precision"] is None else f"{r['precision']:.2f}"
            print(f"  {t:14s} P={p} R={r['recall']:.2f} (tp{r['tp']}/fp{r['fp']}/fn{r['fn']})")

    # ── klipová kalibrace na train doménových souborech ────────────────────
    batch = CedBatch()
    def domain(side):
        return [(i, items[i]) for i in sorted(side)
                if items[i][2] in ("domain_pos", "domain_neg", "bg")]
    print("\nkalibruji…", flush=True)
    S, meta_rows = [], []
    for i, (path, cls, src) in domain(tr_files):
        try:
            S.append(clip_scores(batch, model, scaler, active, path))
            neg_cls = None
            if src == "domain_neg":
                m = re.match(r".+_\d{8}_\d{6}_([a-z_]+)_\d", os.path.basename(path))
                neg_cls = m.group(1) if m else None
            meta_rows.append((cls, src, neg_cls))
        except Exception as e:
            print(f"  ! {os.path.basename(path)}: {e}")
    S = np.stack(S)
    thr = {}
    for ci, cls in enumerate(active):
        pos = np.array([s[ci] for s, (c, _r, _n) in zip(S, meta_rows) if c == cls])
        neg = np.array([s[ci] for s, (c, r, n) in zip(S, meta_rows)
                        if c != cls and (r == "bg" or n == cls)])
        if len(neg) < 20:
            neg = np.array([s[ci] for s, (c, r, _n) in zip(S, meta_rows) if r == "bg"])
        if len(pos) >= 5:
            best = 0.999
            for cand in sorted(np.unique(np.round(pos, 3))):
                tp = int((pos >= cand).sum()); fp = int((neg >= cand).sum())
                if tp + fp and tp / (tp + fp) >= TARGET_PRECISION:
                    best = float(cand); break
            thr[cls] = best
        else:
            thr[cls] = round(min(max(float(np.quantile(neg, NEG_QUANTILE)) + 0.02, 0.5), 0.999), 4)

    # ── held-out deploy verifikace ─────────────────────────────────────────
    fp = keep = npos = nneg = 0
    per_cls = {}
    for i, (path, cls, src) in domain(te_files):
        if src == "bg":
            continue
        try:
            s = clip_scores(batch, model, scaler, active, path)
        except Exception:
            continue
        fired = {active[ci] for ci in range(len(active)) if s[ci] >= thr[active[ci]]}
        if src == "domain_neg":
            nneg += 1
            if fired: fp += 1
        else:
            npos += 1
            d = per_cls.setdefault(cls, [0, 0]); d[1] += 1
            if cls in fired:
                keep += 1; d[0] += 1
    batch.close()
    print(f"\n=== DEPLOY held-out (ced.cpp q8, kalibrováno) ===")
    print(f"  FP míra: {fp}/{nneg} = {100*fp/max(1,nneg):.1f} %   "
          f"recall: {keep}/{npos} = {100*keep/max(1,npos):.0f} %")
    for cls, (k, n) in sorted(per_cls.items()):
        print(f"    {cls:12s}: {k}/{n}")

    # ── export pro betterC: flat f32 blob + meta ───────────────────────────
    import torch
    layers = [m for m in model if isinstance(m, torch.nn.Linear)]
    blob = bytearray()
    dims = []
    blob += scaler.mean_.astype("<f4").tobytes()
    blob += scaler.scale_.astype("<f4").tobytes()
    for lin in layers:
        w = lin.weight.detach().cpu().numpy().astype(np.float32)  # [out,in]
        b = lin.bias.detach().cpu().numpy().astype(np.float32)
        dims.append([int(w.shape[1]), int(w.shape[0])])
        blob += w.tobytes() + b.tobytes()
    open(os.path.join(OUT_DIR, "head_weights.bin"), "wb").write(bytes(blob))
    json.dump({"classes": active, "thresholds": thr, "input_dim": NCLS_CED,
               "layers": dims, "scaler": True, "activation": "relu",
               "output": "sigmoid", "decision": "two-consecutive-windows",
               "backbone": "ced-mini q8 GGUF (ced.cpp)", "sr": SR,
               "win_s": 2.0, "hop_s": 1.0,
               "weights_layout": "scaler.mean[527] scaler.scale[527] then per layer W[out,in] row-major + b[out]"},
              open(os.path.join(OUT_DIR, "head_meta.json"), "w"), indent=2)
    print(f"\nexport: head_weights.bin ({len(blob)//1024} KB) + head_meta.json")
    print("HOTOVO —", OUT_DIR)


if __name__ == "__main__":
    main()
