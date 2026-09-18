"""gpu-cite tagger: summed sparse embeddings → two bidirectional gated affine scans with a
depthwise conv in between → pooled context → BIO tag head, name-part head, type head.

The forward pass is written so that src/cpu.ts and src/shader.wgsl can mirror it op for op:

    e_t   = Σ_i emb[row_t[i]]                                  (padding id 0 contributes 0)
    scan  : a = σ(x Wa + ba), b = x Wb + bb, h_t = a ⊙ h_{t-1} + (1 - a) ⊙ b     (h_0 = 0)
    h1    = [scan_fwd(e) ‖ scan_bwd(e)]                          (2H)
    y     = h1 + relu(dwconv3(h1))                              (2H)
    h2    = [scan_fwd(y) ‖ scan_bwd(y)]                          (2H)
    ctx   = mean_t(h2)                                          (2H)
    g_t   = relu([h2_t ‖ y_t ‖ ctx] W1 + b1)                    (HEAD)
    tags  = g_t Wt + bt          parts = g_t Wp + bp
    type  = relu([mean_t(h2) ‖ max_t(h2)] Wc + bc) Wd + bd

Quantization-aware training fake-quantizes every exported tensor to int6 (symmetric,
per-tensor) with a straight-through estimator, so the exported int6 weights reproduce the
training-time forward exactly (up to float summation order).
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .features import TABLE_ROWS, WIDTH
from .labels import NAMEPARTS, ROLES, TAGS, TYPES

LEVELS = 31


class FakeQuant(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w: Tensor) -> Tensor:  # type: ignore[override]
        scale = w.detach().abs().max() / LEVELS
        scale = torch.clamp(scale, min=1e-8)
        return torch.clamp(torch.round(w / scale), -LEVELS, LEVELS) * scale

    @staticmethod
    def backward(ctx, g: Tensor) -> Tensor:  # type: ignore[override]
        return g


def fq(w: Tensor, enabled: bool) -> Tensor:
    return FakeQuant.apply(w) if enabled else w  # type: ignore[return-value]


def affine_scan(a: Tensor, b: Tensor, mask: Tensor, reverse: bool) -> Tensor:
    """Inclusive scan of h_t = a_t * h_{t-1} + b_t along dim 1 (Hillis–Steele doubling).

    a, b: [B, T, H]; mask: [B, T, 1] (1 for real tokens). Padded positions become the identity
    map (a=1, b=0) so a reversed scan is unaffected by trailing padding.
    """
    a = a * mask + (1.0 - mask)
    b = b * mask
    if reverse:
        a = a.flip(1)
        b = b.flip(1)
    T = a.shape[1]
    A, Bv = a, b
    s = 1
    while s < T:
        A_prev = torch.cat([torch.ones_like(A[:, :s]), A[:, :-s]], dim=1)
        B_prev = torch.cat([torch.zeros_like(Bv[:, :s]), Bv[:, :-s]], dim=1)
        Bv = A * B_prev + Bv
        A = A * A_prev
        s *= 2
    h = Bv
    if reverse:
        h = h.flip(1)
    return h * mask


class ScanLayer(nn.Module):
    def __init__(self, d_in: int, hidden: int):
        super().__init__()
        self.wa_f = nn.Parameter(torch.randn(d_in, hidden) / math.sqrt(d_in))
        self.ba_f = nn.Parameter(torch.zeros(hidden))
        self.wb_f = nn.Parameter(torch.randn(d_in, hidden) / math.sqrt(d_in))
        self.bb_f = nn.Parameter(torch.zeros(hidden))
        self.wa_b = nn.Parameter(torch.randn(d_in, hidden) / math.sqrt(d_in))
        self.ba_b = nn.Parameter(torch.zeros(hidden))
        self.wb_b = nn.Parameter(torch.randn(d_in, hidden) / math.sqrt(d_in))
        self.bb_b = nn.Parameter(torch.zeros(hidden))

    def direction(self, x: Tensor, mask: Tensor, wa: Tensor, ba: Tensor, wb: Tensor, bb: Tensor, reverse: bool, q: bool) -> Tensor:
        a = torch.sigmoid(x @ fq(wa, q) + fq(ba, q))
        b = x @ fq(wb, q) + fq(bb, q)
        return affine_scan(a, (1.0 - a) * b, mask, reverse)

    def forward(self, x: Tensor, mask: Tensor, q: bool) -> Tensor:
        f = self.direction(x, mask, self.wa_f, self.ba_f, self.wb_f, self.bb_f, False, q)
        b = self.direction(x, mask, self.wa_b, self.ba_b, self.wb_b, self.bb_b, True, q)
        return torch.cat([f, b], dim=-1)


class CiteTagger(nn.Module):
    def __init__(self, embed: int = 32, hidden: int = 32, head: int = 48):
        super().__init__()
        self.embed_dim, self.hidden, self.head_dim = embed, hidden, head
        C = 2 * hidden
        self.emb = nn.Embedding(TABLE_ROWS, embed, padding_idx=0)
        nn.init.normal_(self.emb.weight, std=0.3)
        with torch.no_grad():
            self.emb.weight[0].zero_()
        self.scan1 = ScanLayer(embed, hidden)
        self.conv_w = nn.Parameter(torch.randn(3, C) * 0.2)
        self.conv_b = nn.Parameter(torch.zeros(C))
        self.scan2 = ScanLayer(C, hidden)
        self.w1 = nn.Parameter(torch.randn(3 * C, head) / math.sqrt(3 * C))
        self.b1 = nn.Parameter(torch.zeros(head))
        self.wt = nn.Parameter(torch.randn(head, len(TAGS)) / math.sqrt(head))
        self.bt = nn.Parameter(torch.zeros(len(TAGS)))
        self.wp = nn.Parameter(torch.randn(head, len(NAMEPARTS)) / math.sqrt(head))
        self.bp = nn.Parameter(torch.zeros(len(NAMEPARTS)))
        self.wc = nn.Parameter(torch.randn(2 * C, hidden) / math.sqrt(2 * C))
        self.bc = nn.Parameter(torch.zeros(hidden))
        self.wd = nn.Parameter(torch.randn(hidden, len(TYPES)) / math.sqrt(hidden))
        self.bd = nn.Parameter(torch.zeros(len(TYPES)))
        self.trans = nn.Parameter(torch.zeros(len(TAGS), len(TAGS)))
        self.quant = False

    def export_tensors(self) -> dict[str, Tensor]:
        s1, s2 = self.scan1, self.scan2
        return {
            "emb": self.emb.weight,
            "s1f_wa": s1.wa_f, "s1f_ba": s1.ba_f, "s1f_wb": s1.wb_f, "s1f_bb": s1.bb_f,
            "s1b_wa": s1.wa_b, "s1b_ba": s1.ba_b, "s1b_wb": s1.wb_b, "s1b_bb": s1.bb_b,
            "conv_w": self.conv_w, "conv_b": self.conv_b,
            "s2f_wa": s2.wa_f, "s2f_ba": s2.ba_f, "s2f_wb": s2.wb_f, "s2f_bb": s2.bb_f,
            "s2b_wa": s2.wa_b, "s2b_ba": s2.ba_b, "s2b_wb": s2.wb_b, "s2b_bb": s2.bb_b,
            "w1": self.w1, "b1": self.b1, "wt": self.wt, "bt": self.bt, "wp": self.wp, "bp": self.bp,
            "wc": self.wc, "bc": self.bc, "wd": self.wd, "bd": self.bd, "trans": self.trans,
        }

    def forward(self, rows: Tensor, mask: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """rows: [B, T, WIDTH] int64; mask: [B, T] float. Returns (tag logits [B,T,K],
        part logits [B,T,P], type logits [B,C])."""
        q = self.quant
        m = mask.unsqueeze(-1)
        emb = fq(self.emb.weight, q)
        emb = torch.cat([torch.zeros_like(emb[:1]), emb[1:]], dim=0)  # keep padding row at 0
        e = emb[rows].sum(dim=2) * m  # [B, T, E]
        h1 = self.scan1(e, m, q)
        # depthwise conv, kernel 3, zero padding, over the time axis
        cw, cb = fq(self.conv_w, q), fq(self.conv_b, q)
        left = torch.cat([torch.zeros_like(h1[:, :1]), h1[:, :-1]], dim=1)
        right = torch.cat([h1[:, 1:], torch.zeros_like(h1[:, :1])], dim=1)
        conv = left * cw[0] + h1 * cw[1] + right * cw[2] + cb
        y = (h1 + torch.relu(conv)) * m
        h2 = self.scan2(y, m, q)
        n = mask.sum(dim=1, keepdim=True).clamp(min=1.0)  # [B, 1]
        mean = (h2 * m).sum(dim=1) / n  # [B, C]
        mx = (h2 + (m - 1.0) * 1e4).max(dim=1).values  # [B, C]
        ctx = mean.unsqueeze(1).expand(-1, h2.shape[1], -1)
        g = torch.relu(torch.cat([h2, y, ctx], dim=-1) @ fq(self.w1, q) + fq(self.b1, q))
        tags = g @ fq(self.wt, q) + fq(self.bt, q)
        parts = g @ fq(self.wp, q) + fq(self.bp, q)
        c = torch.relu(torch.cat([mean, mx], dim=-1) @ fq(self.wc, q) + fq(self.bc, q))
        ty = c @ fq(self.wd, q) + fq(self.bd, q)
        return tags, parts, ty

    def transitions(self) -> Tensor:
        return fq(self.trans, self.quant)


def crf_nll(emissions: Tensor, trans: Tensor, gold: Tensor, mask: Tensor) -> Tensor:
    """Linear-chain CRF negative log-likelihood, mean over the batch.

    emissions [B,T,K], trans [K,K] (from→to), gold [B,T] long, mask [B,T] float (prefix mask).
    """
    B, T, K = emissions.shape
    alpha = emissions[:, 0]  # [B, K]
    idx = torch.arange(B)
    score = emissions[:, 0].gather(1, gold[:, :1]).squeeze(1)
    for t in range(1, T):
        m = mask[:, t].unsqueeze(-1)
        nxt = torch.logsumexp(alpha.unsqueeze(2) + trans.unsqueeze(0), dim=1) + emissions[:, t]
        alpha = nxt * m + alpha * (1.0 - m)
        step = emissions[idx, t, gold[:, t]] + trans[gold[:, t - 1], gold[:, t]]
        score = score + step * mask[:, t]
    logz = torch.logsumexp(alpha, dim=1)
    return (logz - score).mean()


def constrained_transitions(trans: Tensor) -> Tensor:
    """Add hard BIO constraints (O→I-X, B-X→I-Y, I-X→I-Y forbidden) for Viterbi."""
    K = len(TAGS)
    R = len(ROLES)
    out = trans.clone()
    neg = -1e4
    for to in range(1 + R, K):  # I-Y
        role = to - 1 - R
        out[0, to] = neg
        for frm in range(1, K):
            frm_role = (frm - 1) % R
            if frm_role != role:
                out[frm, to] = neg
    return out


def viterbi(emissions: Tensor, trans: Tensor) -> list[int]:
    """Single-sequence Viterbi, emissions [T,K]. Start may not be I-X."""
    T, K = emissions.shape
    R = len(ROLES)
    score = emissions[0].clone()
    score[1 + R :] = -1e4
    back = torch.zeros(T, K, dtype=torch.long)
    for t in range(1, T):
        cand = score.unsqueeze(1) + trans  # [from, to]
        best, arg = cand.max(dim=0)
        score = best + emissions[t]
        back[t] = arg
    path = [int(score.argmax())]
    for t in range(T - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    path.reverse()
    return path


def count_params(model: CiteTagger) -> int:
    return sum(t.numel() for t in model.export_tensors().values())


__all__ = ["CiteTagger", "affine_scan", "constrained_transitions", "count_params", "crf_nll", "viterbi", "WIDTH"]
