"""Train gpu-tailwind with int6 quantization-aware training through the shared loop.

    uv run python -m gpu_tailwind.train [--epochs 30] [--seed 0] [--minutes 24] [--run default]

Writes training/runs/<run>/{best.pt,last.pt,history.json}. CPU only, 2 threads by default.
"""

from __future__ import annotations

from gpu_utils_training.cli import loop_kwargs, run_dir, training_parser
from gpu_utils_training.loop import train

from .batches import batches, encode, load, loss
from .evaluate import evaluate
from .model import build


def main() -> None:
    ap = training_parser("Train gpu-tailwind", epochs=30, lr=4e-3, batch=128, minutes=24.0)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N training examples")
    args = ap.parse_args()
    train_set = encode(load("train", args.limit))
    dev_set = encode(load("heldout", 3000))
    model = build()
    print(f"parameters: {model.parameter_count():,}  train: {len(train_set)}  dev: {len(dev_set)}")
    result = train(
        model,
        lambda _epoch, rng: batches(train_set, args.batch, rng, model.padding_id),
        lambda m: evaluate(m, dev_set),
        loss=loss,
        out_dir=run_dir(__file__, args),
        select="token_acc",
        **loop_kwargs(args),
    )
    print(f"best: {result['best']}  ({result['minutes']} min)")


if __name__ == "__main__":
    main()
