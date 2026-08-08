"""Turn the QAT checkpoint into the one artifact both engines read.

Everything downstream — the Python integer reference and the JavaScript engine
— consumes the same exported bytes. That is deliberate: if the two of them
loaded weights independently, a parity failure would be ambiguous between "the
kernels disagree" and "the loaders disagree", and only the first is interesting.

Emits:
    reference/model_int8.npz     for the Python reference
    dist/model.js                the same data, base64, as an ES module

Weight layouts are chosen for the JavaScript kernels, not for PyTorch:
    stem 3x3    [outC][kh][kw][inC]
    depthwise   [kh][kw][C]        channel-minor, contiguous inner loop
    pointwise   [outC][inC]
    fc          [inC]
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch

from common import CKPT, DIST, INPUT_SIZE, REFDIR, RESULTS
from reference.fixedpoint import quantize_multiplier
from train.model import BLOCKS, ConvBNReLU, Student


def b64_i8(a) -> str:
    """Pack a small non-negative array as base64 uint8.

    Shift amounts are 0-31. As base64 int32 they cost 5.3 characters each,
    which is worse than just writing the decimal — the packing only pays off
    for the wide values.
    """
    arr = np.asarray(a, dtype=np.int64)
    assert arr.min() >= 0 and arr.max() < 256, "shift out of uint8 range"
    return base64.b64encode(arr.astype(np.uint8).tobytes()).decode()


def b64_i32(a) -> str:
    """Pack an int32 array as base64.

    The per-channel bias/multiplier/shift triples are ~1,600 values each. As
    JSON decimals they cost about 33 KB of the bundle; as base64 int32 they
    cost 8 KB, which is most of the difference between comfortably inside the
    page-weight budget and right on its edge.
    """
    return base64.b64encode(np.asarray(a, dtype=np.int32).tobytes()).decode()


def quantize_weight_np(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-output-channel symmetric int8. w is [outC, ...]."""
    flat = w.reshape(w.shape[0], -1)
    amax = np.maximum(np.abs(flat).max(axis=1), 1e-8)
    scale = amax / 127.0
    q = np.rint(w / scale.reshape([-1] + [1] * (w.ndim - 1)))
    q = np.clip(q, -127, 127).astype(np.int8)
    return q, scale


def build_layers(model: Student) -> tuple[list[dict], list[np.ndarray]]:
    layers: list[dict] = []
    blobs: list[np.ndarray] = []

    # The observer stores its range in float32, so the round-tripped value is
    # 1/255 to float32 precision rather than to double precision. Check at that
    # tolerance, then use the exact rational: the browser's input tensor *is*
    # the raw pixel byte, so the scale it implies is exactly 1/255 and the
    # reference must use the same number.
    s_in_stored = float(model.input_q.scale)
    assert abs(s_in_stored - 1.0 / 255.0) < 1e-8, (
        f"input scale must be 1/255 so the browser can pass raw canvas bytes "
        f"straight in; got {s_in_stored}"
    )
    s_in = 1.0 / 255.0

    H = W = INPUT_SIZE

    def emit(mod: ConvBNReLU, kind: str, s_in: float, H: int, W: int):
        wf, bf = mod.folded()
        wf = wf.detach().cpu().numpy().astype(np.float64)
        bf = bf.detach().cpu().numpy().astype(np.float64)
        q, s_w = quantize_weight_np(wf)
        s_out = float(mod.aq.scale)
        # A collapsed activation range means the observer never saw signal
        # through this layer. Everything downstream still exports "fine" — the
        # scales just become absurd and the biases overflow int32 somewhere
        # much later, where the cause is no longer visible.
        assert s_out > 1e-6, (
            f"{kind} layer activation range collapsed (scale {s_out:.3g}); "
            f"the calibration pass saw no signal through this layer"
        )

        stride = mod.conv.stride[0]
        outC = wf.shape[0]

        if kind == "conv":                       # [outC,inC,kh,kw] -> [outC,kh,kw,inC]
            packed = q.transpose(0, 2, 3, 1).reshape(-1)
            inC = wf.shape[1]
        elif kind == "dw":                       # [C,1,kh,kw] -> [kh,kw,C]
            packed = q[:, 0].transpose(1, 2, 0).reshape(-1)
            inC = outC
        else:                                    # pointwise [outC,inC,1,1] -> [outC,inC]
            packed = q[:, :, 0, 0].reshape(-1)
            inC = wf.shape[1]

        bias_q = np.rint(bf / (s_w * s_in)).astype(np.int64)
        assert np.abs(bias_q).max() < 2 ** 31, "int32 bias overflow"

        m = (s_w * s_in) / s_out
        assert m.max() < 1.0, (
            f"multiplier >= 1 in a {kind} layer (max {m.max():.4f}); the "
            f"kernels only implement right shifts"
        )
        m0, sh = zip(*(quantize_multiplier(float(v)) for v in m))

        outH, outW = -(-H // stride), -(-W // stride)
        layers.append({
            "kind": kind,
            "inH": H, "inW": W, "inC": inC,
            "outH": outH, "outW": outW, "outC": outC,
            "stride": stride,
            "wOffset": int(sum(b.size for b in blobs)),
            "wLen": int(packed.size),
            "bias": b64_i32(bias_q),
            "m0": b64_i32(m0),
            "shift": b64_i8(sh),
            "scaleOut": s_out,
        })
        blobs.append(packed)
        return s_out, outH, outW

    s_in, H, W = emit(model.stem, "conv", s_in, H, W)
    for i in range(0, len(model.blocks), 2):
        s_in, H, W = emit(model.blocks[i], "dw", s_in, H, W)
        s_in, H, W = emit(model.blocks[i + 1], "pw", s_in, H, W)

    # Global average pool: input and output share a scale, so the only
    # multiplier is 1/(H*W), applied through the same fixed-point path.
    pool_m0, pool_shift = quantize_multiplier(1.0 / (H * W))
    pool = {"hw": H * W, "C": BLOCKS[-1][0], "m0": pool_m0, "shift": pool_shift}

    # Final linear layer. Kept in int32: the sign of the accumulator is the
    # decision, and a positive scale cannot change a sign.
    fc_w = model.fc.weight.detach().cpu().numpy().astype(np.float64)
    fc_b = float(model.fc.bias.detach().cpu().numpy()[0])
    q_fc, s_fc = quantize_weight_np(fc_w)
    s_fc = float(s_fc[0])
    fc_bias_q = int(round(fc_b / (s_fc * s_in)))

    # Fold the calibrated decision threshold into the bias, so the deployed
    # model's decision is plainly `accumulator > 0` and the browser carries no
    # knowledge of the class imbalance the sampler introduced. Absent a
    # calibration file this is a no-op and the threshold stays at zero.
    cal_path = RESULTS / "calibration.json"
    threshold = 0
    if cal_path.exists():
        threshold = int(json.loads(cal_path.read_text())["threshold_int32"])
        fc_bias_q -= threshold

    # Always checked, not just when a calibration file exists. The head bias is
    # where a bad activation scale anywhere upstream finally shows up, and the
    # int32 accumulator in both engines has no room to absorb it.
    assert abs(fc_bias_q) < 2 ** 31, (
        f"head bias {fc_bias_q} overflows int32 — an activation scale upstream "
        f"is wrong (last layer scale {s_in:.3g})"
    )

    head = {
        "inC": int(fc_w.shape[1]),
        "bias": fc_bias_q,
        "calibratedThreshold": threshold,
        "wOffset": int(sum(b.size for b in blobs)),
        "wLen": int(q_fc.size),
        # Display only, applied after the decision has already been made.
        "logitScale": s_fc * s_in,
    }
    blobs.append(q_fc.reshape(-1))
    return layers, blobs, pool, head


def main() -> None:
    ckpt = CKPT / "student_qat.pt"
    if not ckpt.exists():
        sys.exit(f"missing {ckpt} — run train/distill.py first")

    model = Student()
    for m in model.modules():
        if isinstance(m, ConvBNReLU):
            m.fold_bn = True
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()

    layers, blobs, pool, head = build_layers(model)
    weights = np.concatenate(blobs).astype(np.int8)

    spec = {
        "inputSize": INPUT_SIZE,
        "inputScale": 1.0 / 255.0,
        "layers": layers,
        "pool": pool,
        "head": head,
        "weightBytes": int(weights.size),
    }

    np.savez_compressed(
        REFDIR / "model_int8.npz",
        weights=weights,
        spec=np.frombuffer(json.dumps(spec).encode(), dtype=np.uint8),
    )

    b64 = base64.b64encode(weights.tobytes()).decode()
    js = (
        "// Generated by train/export.py. Do not edit.\n"
        "// int8 weights, per-output-channel symmetric, zero-point 0.\n"
        f"export const SPEC = {json.dumps(spec)};\n"
        f'export const WEIGHTS_B64 = "{b64}";\n'
    )
    (DIST / "model.js").write_text(js)

    n_params = int(weights.size)
    print(f"layers:        {len(layers)}")
    print(f"weight bytes:  {n_params:,} (int8)")
    print(f"base64 size:   {len(b64):,} chars ({len(b64) / 1024:.1f} KB)")
    print(f"model.js:      {(DIST / 'model.js').stat().st_size / 1024:.1f} KB")
    print(f"wrote {REFDIR / 'model_int8.npz'} and {DIST / 'model.js'}")


if __name__ == "__main__":
    main()
