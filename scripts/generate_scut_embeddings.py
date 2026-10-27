"""Generate filename-aligned ArcFace embeddings for SCUT-FBP5500."""

import argparse
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data_preparation.datasets.scut import SCUT_CACHE_FILE, load_scut_metadata
from data_preparation.embeddings import generate_scut_embeddings


DEFAULT_DATA_DIR = REPO_ROOT / "data" / "SCUT-FBP5500_v2"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        parser.error(f"SCUT data directory not found: {args.data_dir}")

    metadata, _ = load_scut_metadata(args.data_dir)
    filenames = metadata["filename"].to_numpy()
    image_paths = [args.data_dir / "Images" / name for name in filenames]
    output = args.data_dir / SCUT_CACHE_FILE

    result = generate_scut_embeddings(image_paths, filenames, output)
    print(f"SCUT embeddings: {result['matrix'].shape} -> {output}")

    diagnostics = result.get("diagnostics")
    if diagnostics:
        print(f"Padding used: {diagnostics['padding_used']}")
        print(f"Images with multiple detected faces: {diagnostics['multiple_face_images']}")
        print(f"Images without embeddings: {len(diagnostics['failures'])}")


if __name__ == "__main__":
    main()
