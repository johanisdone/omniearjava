"""Export a trained head (.head_train/head.npz) to models/onnx/head.onnx.

Contract mirrors the shipped ShantyCam head so CedMiniThreatModel is unchanged:
    input  "logits"        float32[b, 527]
    output "probabilities" float32[b, 17]

The StandardScaler mean/scale is folded into the first weight layer. Outputs are
the 7 trained sigmoid probabilities scattered into their class columns; the
other 10 columns are exactly 0.

Run with the reference interpreter (has onnx + onnxruntime):
    reference/.venv-ref/Scripts/python.exe scripts/export_head_onnx.py
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort


def fold_first(W, b, mean, scale):
    Wf = (W / scale[:, None]).astype(np.float32)
    bf = (b - (W * (mean / scale)[:, None]).sum(0)).astype(np.float32)
    return Wf, bf


def build(head_npz, n_classes):
    h = np.load(head_npz, allow_pickle=True)
    used = [str(x) for x in h["used"]]
    used_idx = [int(x) for x in h["used_idx"]]
    mean = h["mean"].astype(np.float32)
    scale = h["scale"].astype(np.float32)
    n_in = mean.shape[0]

    models = []
    if str(h["kind"]) == "linear":
        models.append([(h["W"].T.copy(), h["b"].copy())])  # (in,out),(out,)
    else:
        nm = int(h["nmodels"])
        for mi in range(nm):
            nl = int(h[f"nlayers{mi}"])
            models.append([(h[f"W{mi}_{li}"].astype(np.float32),
                            h[f"b{mi}_{li}"].astype(np.float32)) for li in range(nl)])

    inits, nodes = [], []

    outs = []
    for mi, layers in enumerate(models):
        xin = "logits"
        for li, (W, b) in enumerate(layers):
            tag = f"m{mi}_l{li}"
            if li == 0:
                Wf, bf = fold_first(W, b, mean, scale)
                inits.append(numpy_helper.from_array(Wf.copy(), f"{tag}_W"))
                inits.append(numpy_helper.from_array(bf, f"{tag}_b"))
            else:
                inits.append(numpy_helper.from_array(W.astype(np.float32).copy(), f"{tag}_W"))
                inits.append(numpy_helper.from_array(b.astype(np.float32), f"{tag}_b"))
            nodes.append(helper.make_node("MatMul", [xin, f"{tag}_W"], [f"{tag}_mm"]))
            nodes.append(helper.make_node("Add", [f"{tag}_mm", f"{tag}_b"], [f"{tag}_z"]))
            last = li == len(models[mi]) - 1
            act = "Sigmoid" if last else "Relu"
            nodes.append(helper.make_node(act, [f"{tag}_z"], [f"{tag}_a"]))
            xin = f"{tag}_a"
        outs.append(xin)

    if len(outs) == 1:
        probs7 = outs[0]
    else:
        s = outs[0]
        for k, o in enumerate(outs[1:], 1):
            nodes.append(helper.make_node("Add", [s, o], [f"sum{k}"]))
            s = f"sum{k}"
        inits.append(numpy_helper.from_array(np.float32(len(outs)), "ndiv"))
        nodes.append(helper.make_node("Div", [s, "ndiv"], ["probs7"]))
        probs7 = "probs7"

    sel = np.zeros((len(used), n_classes), dtype=np.float32)
    bias = np.zeros((n_classes,), dtype=np.float32)
    for k, ci in enumerate(used_idx):
        sel[k, ci] = 1.0
    inits.append(numpy_helper.from_array(sel, "sel"))
    inits.append(numpy_helper.from_array(bias, "obias"))
    nodes.append(helper.make_node("MatMul", [probs7, "sel"], ["probs_scatter"]))
    nodes.append(helper.make_node("Add", ["probs_scatter", "obias"], ["probabilities"]))

    graph = helper.make_graph(
        nodes, "omniear_head",
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, ["b", n_in])],
        [helper.make_tensor_value_info("probabilities", TensorProto.FLOAT, ["b", n_classes])],
        inits)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    return model, h


def reference_numpy(head_npz, X):
    h = np.load(head_npz, allow_pickle=True)
    mean, scale = h["mean"], h["scale"]
    xs = (X - mean) / scale
    if str(h["kind"]) == "linear":
        W, b = h["W"], h["b"]
        z = xs @ W.T + b
        p = 1 / (1 + np.exp(-z))
    else:
        acc = None
        for mi in range(int(h["nmodels"])):
            x = xs
            nl = int(h[f"nlayers{mi}"])
            for li in range(nl):
                x = x @ h[f"W{mi}_{li}"] + h[f"b{mi}_{li}"]
                x = 1 / (1 + np.exp(-x)) if li == nl - 1 else np.maximum(x, 0)
            acc = x if acc is None else acc + x
        p = acc / int(h["nmodels"])
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", default=".head_train/head.npz", type=Path)
    ap.add_argument("--out", default="models/onnx/head.onnx", type=Path)
    ap.add_argument("--n-classes", type=int, default=17)
    ap.add_argument("--no-install", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    head_npz = root / args.head
    model, meta = build(head_npz, args.n_classes)
    onnx.checker.check_model(model)

    tmp = root / args.out.with_suffix(".onnx.new")
    onnx.save(model, str(tmp))

    sess = ort.InferenceSession(str(tmp), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(0)
    X = rng.standard_normal((4, meta["mean"].shape[0])).astype(np.float32)
    got = sess.run(["probabilities"], {"logits": X})[0]
    ref = reference_numpy(head_npz, X).astype(np.float32)
    used_idx = [int(x) for x in meta["used_idx"]]
    err = float(np.max(np.abs(got[:, used_idx] - ref)))
    print(f"onnx-vs-numpy max abs err (used cols) = {err:.3e}")
    assert err < 1e-4, "export mismatch"

    if args.no_install:
        print(f"built {tmp} (not installed)")
        return
    dst = root / args.out
    if dst.exists():
        shutil.copy2(dst, dst.with_suffix(".onnx.bak"))
    data = dst.with_suffix(".onnx.data")
    if data.exists():
        shutil.copy2(data, data.with_suffix(".bak"))
        data.unlink()
    shutil.move(str(tmp), str(dst))
    print(f"installed {dst}")


if __name__ == "__main__":
    main()
