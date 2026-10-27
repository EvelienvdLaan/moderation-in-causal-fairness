"""Generate row-aligned MiniLM embeddings for the Reddit dataset."""

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data_preparation.embeddings import generate_reddit_embeddings


DEFAULT_INPUT = REPO_ROOT / "data" / "reddit" / "2018.json"
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "data"
    / "reddit"
    / "2018_paraphrase_multilingual_minilm_l12_v2_embeddings.npy"
)


def read_bodies(path):
    """Read every post body without changing the source row order."""
    bodies = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            body = json.loads(line).get("body")
            bodies.append("" if body is None else str(body))
    return bodies


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    if not args.input.is_file():
        parser.error(f"Reddit data not found: {args.input}")

    bodies = read_bodies(args.input)
    result = generate_reddit_embeddings(
        bodies,
        args.output,
        batch_size=args.batch_size,
    )
    print(f"Reddit embeddings: {result['matrix'].shape} -> {args.output}")


if __name__ == "__main__":
    main()
