#!/usr/bin/env python3
"""AudioDet v4.2-distill — knowledge distillation z PaSST v3 (učitel) do CED hlavy (žák).

Motivace: žák (CED-mini q8 hlava, hard labely) má na doménovém deploy held-outu
recall ~79 %/FP ~25 %; učitel PaSST v3 je přesnější (~91 %/10 %). Backbone CED je
dobrý (AudioSet mAP 0.49 ≥ PaSST 0.47) — slabá je HLAVA. Destilujeme měkké cíle
učitele do CED hlavy: α·KD(žák‖učitel, teplota T) + (1−α)·BCE(hard).

Klíčová volba UČITELE (empiricky ověřeno, viz sanity):
  v3 hlava byla trénována na 5s oknech PaSST s MAX-poolingem (train.extract_embedding).
  Holé 2s okno je pro ni OUT-OF-DISTRIBUTION (siren 2s→0.0 vs file-5s→1.0). Proto
  měkký cíl počítáme PER CROP na 5s okně ZAROVNANÉM (centrovaném) na daný 2s výřez
  → PaSST 527 logity → v3 hlava → 15 pravděpodobností. Augmentované výřezy sdílí
  čistý cíl svého (path, off). Cache: ~/audiodet_review/emb_v3_soft.pkl, klíč (path,off).

Žák trénuje na CED q8 reprezentaci (emb_v4ced.pkl, identické crop klíče jako v4.2).
Eval JE IDENTICKÝ s train_v4_ced: crop held-out prf + deploy-sim (2s/1s sliding,
klipová kalibrace P≥0.9 s pravidlem dvou po sobě jdoucích oken, per-class negativa).

Grid: T∈{2,3,4}, α∈{0.5,0.7,0.9}; výběr dle DOMÉNOVÉHO held-out F1. Baseline (α=0,
čistě hard) se přetrénuje ve STEJNÉM běhu ⇒ srovnání v4.2 vs distill je apples-to-apples.

Export (jen když distill pomůže): model_v4_distill/head_weights.bin + head_meta.bin
(+ .json) ve STEJNÉM formátu jako model_v4_ced. Kopie do ~/audiodet_review/model_v4_distill/.
"""
import contextlib, glob, json, os, pickle, re, struct, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_v2 import predict, pick_thresholds, prf  # noqa: E402
from train_v4 import (REVIEW, SEED, SR, WIN, HOP, AUG_CLASSES,  # noqa: E402
                      POS_CROPS, NEG_CROPS, AUG_CROPS, V4_TYPES, Augmenter,
                      collect_files, file_split, crops_for, crop, load16)
from train_v4_ced import (CedBatch, build_plan, extract, CACHE_V4, CACHE_CED,  # noqa: E402
                          NCLS_CED, TARGET_PRECISION, NEG_QUANTILE)
import random  # noqa: E402

OUT_DIR = os.path.join(REVIEW, "model_v4_distill")
TEMPLATE_DIR = os.path.join(REVIEW, "model_v4_ced")     # zdroj kind-enumu pro SBAH
CACHE_SOFT = os.path.join(REVIEW, "emb_v3_soft.pkl")    # učitelovy měkké cíle (path,off)
V3_DIR = os.path.join(REVIEW, "model_v3")
SR32 = 32000
TEACH_WIN = 5 * SR32          # 5s okno učitele (v3 nativní režim)
GRID_T = [2.0, 3.0, 4.0]
GRID_ALPHA = [0.5, 0.7, 0.9]
EPOCHS = 1000
_NULL = open(os.devnull, "w")


# ── učitel: PaSST → v3 hlava ────────────────────────────────────────────────
def load_teacher():
    import torch
    import onnxruntime as ort
    from hear21passt.base import get_basic_model
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    with contextlib.redirect_stdout(_NULL):
        passt = get_basic_model(mode="logits").eval().to(dev)
    s3 = ort.InferenceSession(os.path.join(V3_DIR, "audiodet_multilabel.onnx"),
                              providers=["CPUExecutionProvider"])
    classes3 = json.load(open(os.path.join(V3_DIR, "audiodet_multilabel.json")))["classes"]
    return passt, dev, s3, s3.get_inputs()[0].name, classes3


def teacher_window(x32, off16):
    """5s okno @32k centrované na 2s výřez [off16, off16+WIN] (16k)."""
    center32 = int((off16 + WIN / 2) / SR * SR32)
    start = center32 - TEACH_WIN // 2
    if len(x32) <= TEACH_WIN:
        w = np.pad(x32, (0, TEACH_WIN - len(x32)))
    else:
        start = max(0, min(start, len(x32) - TEACH_WIN))
        w = x32[start:start + TEACH_WIN]
    return w.astype(np.float32)


def compute_soft_targets(plan, items, tr_files):
    """Pro každé UNIKÁTNÍ (path, off) train výřezu → v3 pravděpodobnosti [15].

    Cache emb_v3_soft.pkl. Augmentované výřezy sdílí (path,off) svého zdroje."""
    cache = pickle.load(open(CACHE_SOFT, "rb")) if os.path.exists(CACHE_SOFT) else {}
    # unikátní (path, off) přes train výřezy (base i aug; aug reuse base offset)
    want = []
    seen = set()
    for k, i, off, aug in plan:
        is_tr = (i in tr_files) or (aug is not None)
        if not is_tr:
            continue
        path = items[i][0]
        key = (path, off)
        if key in seen:
            continue
        seen.add(key)
        if key not in cache:
            want.append((path, off, i))
    print(f"učitel v3: {len(seen)} unikátních (path,off), {len(want)} nových", flush=True)
    if not want:
        return cache

    import torch
    passt, dev, s3, in3, classes3 = load_teacher()

    def head(logits):
        return s3.run(None, {in3: logits.astype(np.float32)})[0]

    import librosa
    wav32 = {}
    B = 32
    t0 = time.time()
    buf_w, buf_key = [], []

    def flush():
        if not buf_w:
            return
        with contextlib.redirect_stdout(_NULL):
            with torch.no_grad():
                lg = passt(torch.from_numpy(np.stack(buf_w)).float().to(dev)).cpu().numpy()
        probs = head(lg)
        for key, p in zip(buf_key, probs):
            cache[key] = p.astype(np.float32)
        buf_w.clear(); buf_key.clear()

    for n, (path, off, i) in enumerate(want):
        if path not in wav32:
            if len(wav32) > 256:
                wav32.clear()
            try:
                a, _ = librosa.load(path, sr=SR32, mono=True)
                wav32[path] = a
            except Exception:
                wav32[path] = np.zeros(TEACH_WIN, dtype=np.float32)
        buf_w.append(teacher_window(wav32[path], off))
        buf_key.append((path, off))
        if len(buf_w) >= B:
            flush()
        if n and n % 2560 == 0:
            el = time.time() - t0
            print(f"  [{n}/{len(want)}] {el:.0f}s, ETA {el/n*(len(want)-n):.0f}s", flush=True)
            pickle.dump(cache, open(CACHE_SOFT, "wb"))
    flush()
    pickle.dump(cache, open(CACHE_SOFT, "wb"))
    return cache


# ── dataset matice (identické s train_v4_ced) ───────────────────────────────
def build_matrices():
    items = collect_files()
    tr_files, te_files = file_split(items)
    plan = build_plan(items, tr_files)
    cache = extract(plan, items)   # emb_v4ced.pkl — CED q8 527 features (cachováno)
    active = [t for t in V4_TYPES if any(c == t for _p, c, _s in items)]
    X, Y, is_tr, keys = [], [], [], []
    for k, i, off, aug in plan:
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
        keys.append((items[i][0], off))     # (path, off) pro učitelův cíl
    X = np.stack(X); Y = np.stack(Y); is_tr = np.array(is_tr)
    return items, tr_files, te_files, plan, active, X, Y, is_tr, keys


def build_soft_matrix(keys, is_tr, active, soft_cache, classes3):
    """Měkké cíle [Ntr, nActive] + maska tříd s učitelem + per-sample validita."""
    tr_idx = np.where(is_tr)[0]
    # mapování učitel(15) → žák(active) sloupce
    col = {c: active.index(c) for c in classes3 if c in active}
    class_mask = np.zeros(len(active), dtype=np.float32)
    for c in col:
        class_mask[col[c]] = 1.0
    soft = np.zeros((len(tr_idx), len(active)), dtype=np.float32)
    valid = np.zeros(len(tr_idx), dtype=np.float32)
    for j, gi in enumerate(tr_idx):
        p = soft_cache.get(keys[gi])
        if p is None:
            continue
        valid[j] = 1.0
        for ci3, c in enumerate(classes3):
            if c in col:
                soft[j, col[c]] = p[ci3]
    n_have = int(class_mask.sum())
    print(f"měkké cíle: {int(valid.sum())}/{len(tr_idx)} train výřezů s učitelem, "
          f"{n_have}/{len(active)} tříd s učitelem "
          f"(bez učitele: {[active[i] for i in range(len(active)) if class_mask[i]==0]})")
    return soft, class_mask, valid, tr_idx


# ── trénink hlavy: baseline (hard) i distill ────────────────────────────────
def fit_scaler(X_tr):
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.fit(X_tr)
    return sc


def train_head(X_tr, Y_tr, scaler, n_classes, *, soft=None, class_mask=None,
               valid=None, T=2.0, alpha=0.0, seed=SEED):
    """MLP 527→512→256→128→N. alpha=0 ⇒ čistě hard BCE (baseline v4.2).

    KD (per-class binární, teplota T): T²·BCE(σ(z_s/T), σ(logit(p_t)/T)) přes
    třídy s učitelem (class_mask) a validní vzorky (valid). Hard: pos_weighted BCE."""
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    Xs = scaler.transform(X_tr)
    X_t = torch.from_numpy(Xs).float().to(dev)
    Y_t = torch.from_numpy(Y_tr).float().to(dev)
    model = nn.Sequential(
        nn.Linear(527, 512), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.3),
        nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(128, n_classes),
    ).to(dev)
    pos_weight = torch.zeros(n_classes, device=dev)
    for i in range(n_classes):
        n_pos = Y_t[:, i].sum()
        pos_weight[i] = (len(Y_t) - n_pos) / (n_pos + 1)
    hard_crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    use_kd = alpha > 0.0 and soft is not None
    if use_kd:
        st = torch.from_numpy(soft).float().to(dev).clamp(1e-4, 1 - 1e-4)
        z_teacher = torch.log(st / (1 - st))                 # inverze sigmoidu → logity
        q_teacher = torch.sigmoid(z_teacher / T)             # měkký cíl při teplotě T
        cmask = torch.from_numpy(class_mask).float().to(dev)  # [N]
        vmask = torch.from_numpy(valid).float().to(dev)       # [B]
        kd_crit = nn.BCEWithLogitsLoss(reduction="none")

    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    model.train()
    for ep in range(EPOCHS):
        opt.zero_grad()
        z = model(X_t)
        loss = hard_crit(z, Y_t)
        if use_kd:
            kd = kd_crit(z / T, q_teacher)                    # [B, N]
            w = (vmask[:, None] * cmask[None, :])
            kd = (kd * w).sum() / (w.sum() + 1e-8) * (T * T)
            loss = alpha * kd + (1 - alpha) * loss
        loss.backward(); opt.step(); sched.step()
    model.eval()
    return model


# ── deploy kalibrace + held-out (identické s train_v4_ced) ──────────────────
def clip_score_from_windows(model, scaler, W):
    """W = [nWin, 527] CED q8 probs → per-class skóre klipu (2-okna pravidlo)."""
    probs = predict(model, scaler, W)
    if len(probs) == 1:
        return probs[0]
    return np.minimum(probs[:-1], probs[1:]).max(axis=0)


def precompute_domain_windows(items, tr_files, te_files):
    """CED q8 sliding-window matice pro doménové soubory — spočítá se JEDNOU."""
    def domain(side):
        return [(i, items[i]) for i in sorted(side)
                if items[i][2] in ("domain_pos", "domain_neg", "bg")]
    files = domain(tr_files) + domain(te_files)
    win = {}
    batch = CedBatch()
    t0 = time.time()
    for n, (i, (path, cls, src)) in enumerate(files):
        if path in win:
            continue
        try:
            x = load16(path)
            offs = list(range(0, max(len(x) - WIN, 0) + 1, HOP)) or [0]
            wins = np.stack([crop(x, o) for o in offs])
            win[path] = batch.run(wins).copy()
        except Exception as e:
            print(f"  ! win {os.path.basename(path)}: {e}")
        if n and n % 500 == 0:
            print(f"  domain windows [{n}/{len(files)}] {time.time()-t0:.0f}s", flush=True)
    batch.close()
    print(f"domain windows: {len(win)} souborů, {time.time()-t0:.0f}s")
    return win


def calibrate(model, scaler, active, items, tr_files, win):
    """Klipové prahy na train-side doméně: P≥0.9, per-class negativa + bg."""
    rows, S = [], []
    for i in sorted(tr_files):
        path, cls, src = items[i]
        if src not in ("domain_pos", "domain_neg", "bg") or path not in win:
            continue
        S.append(clip_score_from_windows(model, scaler, win[path]))
        neg_cls = None
        if src == "domain_neg":
            m = re.match(r".+_\d{8}_\d{6}_([a-z_]+)_\d", os.path.basename(path))
            neg_cls = m.group(1) if m else None
        rows.append((cls, src, neg_cls))
    S = np.stack(S)
    thr = {}
    for ci, cls in enumerate(active):
        pos = np.array([s[ci] for s, (c, _r, _n) in zip(S, rows) if c == cls])
        neg = np.array([s[ci] for s, (c, r, n) in zip(S, rows)
                        if c != cls and (r == "bg" or n == cls)])
        if len(neg) < 20:
            neg = np.array([s[ci] for s, (c, r, _n) in zip(S, rows) if r == "bg"])
        if len(pos) >= 5:
            best = 0.999
            for cand in sorted(np.unique(np.round(pos, 3))):
                tp = int((pos >= cand).sum()); fp = int((neg >= cand).sum())
                if tp + fp and tp / (tp + fp) >= TARGET_PRECISION:
                    best = float(cand); break
            thr[cls] = best
        else:
            thr[cls] = round(min(max(float(np.quantile(neg, NEG_QUANTILE)) + 0.02, 0.5), 0.999), 4) \
                if len(neg) else 0.999
    return thr


def eval_holdout(model, scaler, active, thr, items, te_files, win):
    """Held-out deploy: FP míra (domain_neg), recall (domain_pos), per-class, F1."""
    fp = keep = npos = nneg = 0
    per_cls = {}
    for i in sorted(te_files):
        path, cls, src = items[i]
        if src not in ("domain_pos", "domain_neg") or path not in win:
            continue
        s = clip_score_from_windows(model, scaler, win[path])
        fired = {active[ci] for ci in range(len(active)) if s[ci] >= thr[active[ci]]}
        if src == "domain_neg":
            nneg += 1
            if fired:
                fp += 1
        else:
            npos += 1
            d = per_cls.setdefault(cls, [0, 0]); d[1] += 1
            if cls in fired:
                keep += 1; d[0] += 1
    P = keep / (keep + fp) if (keep + fp) else 0.0
    R = keep / npos if npos else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) else 0.0
    return {"fp": fp, "nneg": nneg, "keep": keep, "npos": npos,
            "per_cls": per_cls, "precision": P, "recall": R, "f1": F1,
            "fp_rate": fp / max(1, nneg)}


# ── export SBAH (kind-enum z template model_v4_ced) ─────────────────────────
def _parse_meta_kinds(path):
    b = open(path, "rb").read()
    assert b[:4] == b"SBAH"
    o = 4
    ver = struct.unpack_from("<I", b, o)[0]; o += 4
    nLayers = struct.unpack_from("<I", b, o)[0]; o += 4
    o += nLayers * 8
    nClasses = struct.unpack_from("<I", b, o)[0]; o += 4
    kinds = []
    for _ in range(nClasses):
        kinds.append(b[o]); o += 1
        o += 4
    return ver, kinds


def export_model(model, scaler, active, thr):
    import torch
    os.makedirs(OUT_DIR, exist_ok=True)
    layers = [m for m in model.cpu() if isinstance(m, torch.nn.Linear)]
    blob = bytearray()
    blob += scaler.mean_.astype("<f4").tobytes()
    blob += scaler.scale_.astype("<f4").tobytes()
    dims = []
    for lin in layers:
        w = lin.weight.detach().cpu().numpy().astype(np.float32)   # [out,in]
        b = lin.bias.detach().cpu().numpy().astype(np.float32)
        dims.append([int(w.shape[1]), int(w.shape[0])])
        blob += w.tobytes() + b.tobytes()
    open(os.path.join(OUT_DIR, "head_weights.bin"), "wb").write(bytes(blob))

    # kind-enum z template (name→kind); klasifikace i dims musí sedět
    tmpl_classes = json.load(open(os.path.join(TEMPLATE_DIR, "head_meta.json")))["classes"]
    _ver, tmpl_kinds = _parse_meta_kinds(os.path.join(TEMPLATE_DIR, "head_meta.bin"))
    name2kind = {c: tmpl_kinds[i] for i, c in enumerate(tmpl_classes)}
    meta = bytearray(b"SBAH")
    meta += struct.pack("<I", 1)
    meta += struct.pack("<I", len(dims))
    for d in dims:
        meta += struct.pack("<II", d[0], d[1])
    meta += struct.pack("<I", len(active))
    for c in active:
        meta += struct.pack("<B", name2kind[c])
        meta += struct.pack("<f", float(thr[c]))
    open(os.path.join(OUT_DIR, "head_meta.bin"), "wb").write(bytes(meta))

    json.dump({"classes": active, "thresholds": thr, "input_dim": NCLS_CED,
               "layers": dims, "scaler": True, "activation": "relu",
               "output": "sigmoid", "decision": "two-consecutive-windows",
               "backbone": "ced-mini q8 GGUF (ced.cpp)", "sr": SR,
               "win_s": 2.0, "hop_s": 1.0, "distill": True,
               "teacher": "PaSST v3 (5s okno, v3 hlava), KD teplota+alpha viz report",
               "weights_layout": "scaler.mean[527] scaler.scale[527] then per layer W[out,in] row-major + b[out]"},
              open(os.path.join(OUT_DIR, "head_meta.json"), "w"), indent=2)
    return len(blob), len(meta)


def print_deploy(tag, m):
    print(f"  [{tag}] FP {m['fp']}/{m['nneg']} = {100*m['fp_rate']:.1f} %   "
          f"recall {m['keep']}/{m['npos']} = {100*m['recall']:.0f} %   "
          f"P={m['precision']:.2f} F1={m['f1']:.3f}", flush=True)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("=== build matrices (CED q8 features) ===", flush=True)
    items, tr_files, te_files, plan, active, X, Y, is_tr, keys = build_matrices()
    tr_idx = np.where(is_tr)[0]; te_idx = np.where(~is_tr)[0]
    print(f"matice: {len(X)} výřezů (train {len(tr_idx)}, eval {len(te_idx)}), tříd {len(active)}")

    # učitelovy měkké cíle
    print("\n=== teacher soft targets (PaSST v3, 5s) ===", flush=True)
    classes3 = json.load(open(os.path.join(V3_DIR, "audiodet_multilabel.json")))["classes"]
    soft_cache = compute_soft_targets(plan, items, tr_files)
    soft, class_mask, valid, tr_idx2 = build_soft_matrix(keys, is_tr, active, soft_cache, classes3)
    assert np.array_equal(tr_idx, tr_idx2)

    scaler = fit_scaler(X[tr_idx])

    print("\n=== precompute domain windows (once) ===", flush=True)
    win = precompute_domain_windows(items, tr_files, te_files)

    def run_config(tag, alpha, T):
        model = train_head(X[tr_idx], Y[tr_idx], scaler, len(active),
                           soft=soft, class_mask=class_mask, valid=valid,
                           T=T, alpha=alpha)
        thr = calibrate(model, scaler, active, items, tr_files, win)
        m = eval_holdout(model, scaler, active, thr, items, te_files, win)
        thr_crop = pick_thresholds(predict(model, scaler, X[tr_idx]), Y[tr_idx], active)
        res_crop = prf(predict(model, scaler, X[te_idx]), Y[te_idx], thr_crop, active)
        return model, thr, m, res_crop

    results = {}
    print("\n=== BASELINE v4.2 (hard, α=0) ===", flush=True)
    base_model, base_thr, base_m, base_crop = run_config("baseline", 0.0, 2.0)
    print_deploy("v4.2", base_m)
    results["baseline"] = base_m

    grid = []
    print("\n=== GRID distill ===", flush=True)
    for alpha in GRID_ALPHA:
        for T in GRID_T:
            tag = f"a{alpha}_T{T}"
            model, thr, m, crop_res = run_config(tag, alpha, T)
            print_deploy(tag, m)
            grid.append((alpha, T, m, model, thr, crop_res))
            results[tag] = m

    # výběr dle doménového held-out F1 (task); tie-break vyšší recall, pak nižší FP
    grid.sort(key=lambda g: (g[2]["f1"], g[2]["recall"], -g[2]["fp_rate"]), reverse=True)
    best_alpha, best_T, best_m, best_model, best_thr, best_crop = grid[0]
    print(f"\n>>> BEST distill: α={best_alpha} T={best_T}  "
          f"F1={best_m['f1']:.3f} recall={100*best_m['recall']:.0f}% "
          f"FP={100*best_m['fp_rate']:.1f}%")

    # srovnávací tabulka v4.2 vs best-distill (per class deploy held-out)
    print("\n=== v4.2 vs v4.2-distill — DEPLOY held-out (per class) ===")
    print(f"{'třída':13s} {'v4.2 keep':>10s} {'distill keep':>13s}")
    allcls = sorted(set(base_m["per_cls"]) | set(best_m["per_cls"]))
    for cls in allcls:
        b = base_m["per_cls"].get(cls, [0, 0]); d = best_m["per_cls"].get(cls, [0, 0])
        print(f"{cls:13s} {b[0]:>4d}/{b[1]:<4d}     {d[0]:>4d}/{d[1]:<4d}")
    print(f"{'CELKEM FP':13s} {base_m['fp']:>4d}/{base_m['nneg']:<4d}     "
          f"{best_m['fp']:>4d}/{best_m['nneg']:<4d}")
    print(f"{'recall':13s} {100*base_m['recall']:>8.0f}%      {100*best_m['recall']:>8.0f}%")
    print(f"{'FP míra':13s} {100*base_m['fp_rate']:>8.1f}%      {100*best_m['fp_rate']:>8.1f}%")
    print(f"{'F1':13s} {base_m['f1']:>9.3f}      {best_m['f1']:>9.3f}")

    # crop-level srovnání (informativní)
    print("\n=== CROP held-out (v4.2 vs distill), recall per class ===")
    for t in active:
        rb = base_crop[t]; rd = best_crop[t]
        if (rb["tp"] + rb["fn"]) or rb["fp"]:
            print(f"  {t:14s} v4.2 R={0 if rb['recall'] is None else rb['recall']:.2f} "
                  f"(fp{rb['fp']})  distill R={0 if rd['recall'] is None else rd['recall']:.2f} "
                  f"(fp{rd['fp']})")

    # rozhodnutí o exportu: distill musí zvednout recall a NEzhoršit F1 výrazně
    improves = (best_m["recall"] > base_m["recall"] + 1e-9) and \
               (best_m["f1"] >= base_m["f1"] - 1e-9)
    report = {
        "baseline": {k: base_m[k] for k in ("fp", "nneg", "keep", "npos", "precision", "recall", "f1", "fp_rate", "per_cls")},
        "best": {"alpha": best_alpha, "T": best_T,
                 **{k: best_m[k] for k in ("fp", "nneg", "keep", "npos", "precision", "recall", "f1", "fp_rate", "per_cls")}},
        "grid": [{"alpha": a, "T": T, **{k: m[k] for k in ("recall", "fp_rate", "f1", "precision")}}
                 for a, T, m, _mo, _th, _cr in
                 sorted(grid, key=lambda g: (g[0], g[1]))],
        "exported": bool(improves),
    }
    if improves:
        nb, nm = export_model(best_model, scaler, active, best_thr)
        print(f"\nEXPORT: head_weights.bin ({nb//1024} KB) + head_meta.bin ({nm} B) + .json → {OUT_DIR}")
        report["export_dir"] = OUT_DIR
    else:
        print("\nNEEXPORTOVÁNO — destilace nezvedla recall bez ztráty F1 (poctivý negativní výsledek).")
    json.dump(report, open(os.path.join(OUT_DIR, "report.json"), "w"), indent=2, default=str)
    print("HOTOVO —", OUT_DIR)


if __name__ == "__main__":
    main()
