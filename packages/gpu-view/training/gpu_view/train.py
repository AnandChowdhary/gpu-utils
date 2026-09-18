"""Train gpu-view with int6 quantization-aware training on the shared scan family.

    uv run python -m gpu_view.train [--samples 160000] [--epochs 14] [--seed 0] [--run default]

Writes training/runs/<run>/{best.pt,last.pt,history.json}. CPU only, 2 threads by
default; the default run takes ~15 minutes.
"""

from __future__ import annotations

from gpu_utils_training.cli import loop_kwargs, run_dir, training_parser
from gpu_utils_training.loop import train

from . import data, features
from .evaluate import evaluate_both
from .model import build


def main() -> None:
    ap = training_parser("Train gpu-view", epochs=14, lr=3e-3, batch=128)
    ap.add_argument("--samples", type=int, default=160000)
    args = ap.parse_args()
    corpora = data.build(args.samples, seed=args.seed)
    model = build()
    print(f"parameters: {model.parameter_count():,}  feature rows: {features.FEATURE_ROWS}  train: {corpora['train']['rows'].shape[0]}")
    result = train(
        model,
        lambda _epoch, rng: data.batches(corpora["train"], args.batch, rng),
        lambda m: evaluate_both(m, corpora),
        loss=data.loss,
        out_dir=run_dir(__file__, args),
        select="dev_sequence_exact",
        **loop_kwargs(args),
    )
    print(f"best: {result['best']}")


if __name__ == "__main__":
    main()
