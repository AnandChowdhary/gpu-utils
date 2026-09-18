"""Train gpu-paste with int6 quantization-aware training through the shared loop.

    uv run python -m gpu_paste.train [--n 160000] [--epochs 8] [--seed 1] [--minutes 18] [--run default]

Writes training/runs/<run>/{best.pt,last.pt,history.json}. CPU only, 2 threads by default;
the default run takes ~8 minutes plus ~90 s of data generation on first use.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from gpu_paste.data import LABELS
from gpu_paste.dataset import Batch, Batches, build
from gpu_paste.evaluate import evaluate_model, summary
from gpu_paste.model import build as build_model
from gpu_utils_training.cli import loop_kwargs, run_dir, training_parser
from gpu_utils_training.loop import train

# Span labels are dominated by O, so down-weight it a little; the learned kinds are roughly
# balanced by the generator.
SPAN_WEIGHT = torch.ones(len(LABELS))
SPAN_WEIGHT[0] = 0.5


def loss(model: nn.Module, batch: Batch) -> Tensor:
    """Span BIO cross-entropy + kind cross-entropy (kind masked with -1 for structured pastes)."""
    rows, mask, labels, kinds, _ = batch
    out = model(rows, mask)
    span, kind = out["tags"], out["pooled"]
    assert kind is not None
    loss_span = F.cross_entropy(span.reshape(-1, span.shape[-1]), labels.reshape(-1), weight=SPAN_WEIGHT, ignore_index=-100)
    if bool((kinds >= 0).any()):
        return loss_span + F.cross_entropy(kind, kinds, ignore_index=-1)
    return loss_span


def main() -> None:
    ap = training_parser("Train gpu-paste", epochs=8, seed=1, batch=64, lr=3e-3, minutes=18)
    ap.add_argument("--n", type=int, default=160_000, help="training examples to generate")
    ap.add_argument("--heldout", type=int, default=8_000, help="held-out examples (seed + 1000)")
    ap.add_argument("--offline", action="store_true", help="skip the public-domain name/company downloads")
    args = ap.parse_args()

    train_set, _ = build(args.n, args.seed, offline=args.offline)
    heldout, _ = build(args.heldout, args.seed + 1000, offline=args.offline)
    model = build_model()
    print(f"parameters: {model.parameter_count():,}  train: {len(train_set)}  held-out: {len(heldout)}")
    result = train(
        model,
        lambda _epoch, rng: Batches(train_set, args.batch, rng, model.padding_id),
        lambda m: summary(evaluate_model(m, heldout)),
        loss=loss,
        out_dir=run_dir(__file__, args),
        select="score",
        **loop_kwargs(args),
    )
    print(f"best: {result['best']}  ({result['minutes']} min)")


if __name__ == "__main__":
    main()
