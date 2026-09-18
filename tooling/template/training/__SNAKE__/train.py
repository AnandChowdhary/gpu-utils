"""Train __NAME__ with int6 quantization-aware training.

    uv run python -m __SNAKE__.train [--epochs 4] [--seed 0] [--minutes 18] [--run default]

Writes training/runs/<run>/{best.pt,last.pt,history.json}. CPU only, 2 threads by default.
"""

from __future__ import annotations

from gpu_utils_training.cli import loop_kwargs, run_dir, training_parser
from gpu_utils_training.loop import train

from .data import batches, generate, loss
from .evaluate import evaluate
from .model import build


def main() -> None:
    args = training_parser("Train __NAME__", epochs=4, lr=3e-3, batch=64).parse_args()
    train_set, dev_set = generate(20_000, args.seed), generate(2_000, args.seed + 1)
    model = build()
    print(f"parameters: {model.parameter_count():,}  train: {len(train_set)}  dev: {len(dev_set)}")
    result = train(
        model,
        lambda _epoch, rng: list(batches(train_set, args.batch, rng, model.padding_id)),
        lambda m: evaluate(m, dev_set),
        loss=loss,
        out_dir=run_dir(__file__, args),
        select="span_f1",
        **loop_kwargs(args),
    )
    print(f"best: {result['best']}")


if __name__ == "__main__":
    main()
