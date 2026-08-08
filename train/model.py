"""The student: a 136,417-parameter depthwise-separable CNN, and its fake-quant
machinery.

Design is driven entirely by what has to be hand-written in JavaScript later.
Every op here maps to one of four kernels. No hard-swish, no squeeze-excite, no
residual connections, no concat. ReLU only.

Quantization scheme
-------------------
weights      int8, per-output-channel symmetric, q in [-127, 127], zero-point 0
activations  uint8, per-tensor symmetric,        q in [0, 255],    zero-point 0
bias         int32 at scale (s_w[c] * s_in)
accumulate   int32

Activation zero-points are 0 rather than asymmetric because every activation in
this network is the output of a ReLU and therefore non-negative — asymmetric
quantization would buy exactly nothing. The payoff is large: with x_zp = 0 the
convolution accumulator is a plain sum of products with no zero-point
correction term, which is one fewer thing to get wrong in a hand-written kernel.

The input follows the same rule: images are fed as pixel/255, so the input
quantizer lands on scale 1/255, zero-point 0, and the quantized input tensor is
bit-identical to the raw RGB bytes off the canvas.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# (out_channels, stride) for each depthwise-separable block, after the stem.
BLOCKS = [
    (32, 1),
    (64, 2),
    (64, 1),
    (128, 2),
    (128, 1),
    (256, 2),
    (256, 1),
]
STEM_CH = 16
INPUT_SIZE = 96


# ── fake quantization ────────────────────────────────────────────────────────

class _RoundSTE(torch.autograd.Function):
    """round() with a straight-through gradient."""

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, g):
        return g


def _round_ste(x: torch.Tensor) -> torch.Tensor:
    return _RoundSTE.apply(x)


def quantize_weight(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-output-channel symmetric int8. Returns (fake-quantized w, scales)."""
    flat = w.reshape(w.shape[0], -1)
    amax = flat.abs().amax(dim=1).clamp(min=1e-8)
    scale = amax / 127.0
    shape = [-1] + [1] * (w.dim() - 1)
    s = scale.reshape(shape)
    q = _round_ste(w / s).clamp(-127, 127)
    return q * s, scale


class ActObserver(nn.Module):
    """Per-tensor symmetric uint8 observer with EMA range tracking.

    In `observe` mode it only records the range. In `fake` mode it also rounds
    the tensor to the uint8 grid, so QAT sees the error the deployed model will
    actually make.
    """

    def __init__(self, momentum: float = 0.99):
        super().__init__()
        self.register_buffer("amax", torch.tensor(0.0))
        self.register_buffer("primed", torch.tensor(0.0))
        self.momentum = momentum
        self.mode = "off"  # off | observe | fake
        self.frozen = False

    def freeze_at(self, amax: float) -> None:
        """Pin the range instead of observing it.

        Used for the input, which is pinned to 1.0 so the scale is exactly
        1/255 and the quantized input tensor is bit-identical to the raw RGB
        bytes coming off the canvas. If this were left to the observer it would
        settle near 1/255 but not on it, and the browser would have to rescale
        every pixel before the first convolution.
        """
        self.amax.fill_(amax)
        self.primed.fill_(1.0)
        self.frozen = True

    @property
    def scale(self) -> torch.Tensor:
        return self.amax.clamp(min=1e-8) / 255.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "off":
            return x
        if self.training and not self.frozen:
            cur = x.detach().amax()
            if self.primed.item() == 0:
                self.amax.copy_(cur)
                self.primed.fill_(1.0)
            else:
                self.amax.mul_(self.momentum).add_(cur * (1 - self.momentum))
        if self.mode == "observe":
            return x
        s = self.scale
        return _round_ste(x / s).clamp(0, 255) * s


# ── layers ───────────────────────────────────────────────────────────────────

class ConvBNReLU(nn.Module):
    """Conv -> BN -> ReLU -> activation quantizer.

    `groups == in_ch` makes this the depthwise kernel; `k == 1` the pointwise
    kernel; the stem is the only dense 3x3.

    BatchNorm is folded into the convolution *during* QAT, not merely at export.
    Folding only at export would mean the weights QAT trained on are not the
    weights that get quantized — the fold rescales each output channel by
    gamma/sqrt(var+eps), which changes the per-channel max and therefore changes
    every weight scale. Training against the unfolded weights and shipping the
    folded ones is a silent accuracy leak.
    """

    def __init__(self, cin: int, cout: int, k: int, stride: int, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, stride, k // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.aq = ActObserver()
        self.quant_weights = False
        self.fold_bn = False

    def folded(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Conv+BN collapsed into a single weight/bias pair."""
        s = self.bn.weight / torch.sqrt(self.bn.running_var + self.bn.eps)
        w = self.conv.weight * s.reshape(-1, 1, 1, 1)
        b = self.bn.bias - self.bn.running_mean * s
        return w, b

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fold_bn:
            w, b = self.folded()
            if self.quant_weights:
                w, _ = quantize_weight(w)
            x = F.conv2d(x, w, b, self.conv.stride, self.conv.padding,
                         self.conv.dilation, self.conv.groups)
        else:
            w = self.conv.weight
            if self.quant_weights:
                w, _ = quantize_weight(w)
            x = F.conv2d(x, w, None, self.conv.stride, self.conv.padding,
                         self.conv.dilation, self.conv.groups)
            x = self.bn(x)
        x = F.relu(x)
        return self.aq(x)


class Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_q = ActObserver()
        # Images arrive as pixel/255, so the input range is exactly [0, 1].
        self.input_q.freeze_at(1.0)
        self.stem = ConvBNReLU(3, STEM_CH, 3, 2)

        layers: list[nn.Module] = []
        cin = STEM_CH
        for cout, stride in BLOCKS:
            layers.append(ConvBNReLU(cin, cin, 3, stride, groups=cin))  # depthwise
            layers.append(ConvBNReLU(cin, cout, 1, 1))                  # pointwise
            cin = cout
        self.blocks = nn.ModuleList(layers)
        self.pool_q = ActObserver()
        self.fc = nn.Linear(cin, 1)
        self.quant_weights = False

    # -- mode switches -------------------------------------------------------

    def set_quant(self, mode: str, weights: bool) -> None:
        """mode in {off, observe, fake}; weights toggles weight fake-quant."""
        for m in self.modules():
            if isinstance(m, ActObserver):
                m.mode = mode
            if isinstance(m, ConvBNReLU):
                m.quant_weights = weights
        self.quant_weights = weights

    # -- forward -------------------------------------------------------------

    def forward(self, x: torch.Tensor, taps: list | None = None) -> torch.Tensor:
        x = self.input_q(x)
        x = self.stem(x)
        if taps is not None:
            taps.append(x)
        for i, layer in enumerate(self.blocks):
            x = layer(x)
            if taps is not None and i % 2 == 1:
                taps.append(x)
        x = x.mean(dim=(2, 3))
        x = self.pool_q(x)
        w = self.fc.weight
        if self.quant_weights:
            w, _ = quantize_weight(w)
        return F.linear(x, w, self.fc.bias).squeeze(1)


def param_count(model: nn.Module) -> int:
    """Weights that actually ship: convs + fc. BN folds into the convs at export."""
    n = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            n += m.weight.numel()
        elif isinstance(m, nn.Linear):
            n += m.weight.numel() + (m.bias.numel() if m.bias is not None else 0)
    return n


if __name__ == "__main__":
    m = Student()
    x = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)
    y = m(x)
    print("output", tuple(y.shape))
    print("shipped params:", param_count(m))
    print("total params (incl. BN):", sum(p.numel() for p in m.parameters()))
