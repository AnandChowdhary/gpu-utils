"""Train gpu-log with int6 quantization-aware training.

    uv run python -m gpu_log.train [--epochs 5] [--seed 1] [--minutes 18] [--run default]

Default: 160K synthetic lines, 5 epochs, QAT from epoch 1, ~12 minutes on 2 CPU threads.
Writes training/runs/<run>/{best.pt,last.pt,history.json}; then `python -m gpu_log.export`.
"""

from __future__ import annotations

from gpu_utils_training.cli import loop_kwargs, run_dir, training_parser
from gpu_utils_training.loop import train

from .data import batches, cached, loss
from .evaluate import evaluate_dataset
from .model import build


def main() -> None:
    ap = training_parser("Train gpu-log", epochs=5, seed=1, lr=2e-3, batch=48, minutes=18)
    ap.add_argument("--lines", type=int, default=160_000)
    args = ap.parse_args()
    train_set = cached("train", args.lines, 1)
    dev_set = cached("heldout", 12_000, 2)
    model = build()
    print(f"parameters: {model.parameter_count():,}  train: {len(train_set)} lines  dev: {len(dev_set)} lines")
    result = train(
        model,
        lambda _epoch, rng: batches(train_set, args.batch, rng, model.padding_id),
        lambda m: evaluate_dataset(m, dev_set, limit=3000)["flat"],
        loss=loss,
        out_dir=run_dir(__file__, args),
        select="span_f1",
        **loop_kwargs(args),
    )
    print(f"best: {result['best']}  ({result['minutes']} min)")


if __name__ == "__main__":
    main()
