"""Strict loading and row alignment for cached embeddings."""

from pathlib import Path

import numpy as np
import pandas as pd


def load_embeddings(path, *, matrix_key="embeddings", identifiers_key="filenames",
                    mmap=False):
    """Load a finite two-dimensional NPY or NPZ embedding cache."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    cache = np.load(path, mmap_mode="r" if mmap and path.suffix == ".npy" else None,
                    allow_pickle=False)
    if isinstance(cache, np.ndarray):
        matrix, identifiers = cache, None
    else:
        matrix = cache[matrix_key]
        identifiers = cache[identifiers_key].astype(str) if identifiers_key in cache else None
    if matrix.ndim != 2 or not np.issubdtype(matrix.dtype, np.number):
        raise ValueError("Embeddings must be a numeric two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("Embeddings contain non-finite values")
    if identifiers is not None and (len(identifiers) != len(matrix) or len(set(identifiers)) != len(identifiers)):
        raise ValueError("Embedding identifiers must be aligned and unique")
    return {"matrix": matrix, "identifiers": identifiers, "source": str(path)}


def attach_embeddings(data, embeddings, *, row_ids=None, data_id=None,
                      prefix="embedding"):
    """Attach embeddings by verified identifier or explicit row order."""
    matrix = embeddings["matrix"]
    identifiers = embeddings.get("identifiers")
    if data_id is not None:
        if identifiers is None:
            raise ValueError("The embedding cache has no identifiers")
        lookup = {value: index for index, value in enumerate(identifiers)}
        keys = data[data_id].astype(str).to_numpy()
        absent = [key for key in keys if key not in lookup]
        if absent:
            raise ValueError(f"Rows lack embeddings; first identifiers: {absent[:5]}")
        matrix = matrix[[lookup[key] for key in keys]]
    elif row_ids is not None:
        matrix = matrix[np.asarray(row_ids, dtype=int)]
    elif len(matrix) != len(data):
        raise ValueError("Row-order attachment requires equal row counts")
    columns = tuple(f"{prefix}_{index:03d}" for index in range(matrix.shape[1]))
    embedded = pd.concat([
        data.reset_index(drop=True).copy(),
        pd.DataFrame(np.asarray(matrix), columns=columns),
    ], axis=1)
    return embedded, columns


def generate_reddit_embeddings(texts, cache_path, *,
                               model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                               batch_size=32, encoder=None):
    """Load cached Reddit embeddings or generate them in supplied row order."""
    texts = [str(text) for text in texts]
    cache_path = Path(cache_path).expanduser()
    if cache_path.is_file():
        cached = load_embeddings(cache_path)
        if len(cached["matrix"]) != len(texts):
            raise ValueError("Cached Reddit embeddings do not match the supplied text rows")
        return cached
    if encoder is None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as error:
            raise ImportError("Generating Reddit embeddings requires sentence-transformers") from error
        encoder = SentenceTransformer(model_name)
    matrix = np.asarray(encoder.encode(
        texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True,
    ))
    if matrix.ndim != 2 or len(matrix) != len(texts) or not np.isfinite(matrix).all():
        raise ValueError("Generated Reddit embeddings are invalid or misaligned")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, matrix)
    return {"matrix": matrix, "identifiers": None, "source": str(cache_path)}


def _neutral_padding(image, fraction, cv2):
    """Add the notebook's symmetric gray padding to an image."""
    if fraction == 0:
        return image
    height, width = image.shape[:2]
    pad_y = max(1, int(round(height * fraction)))
    pad_x = max(1, int(round(width * fraction)))
    return cv2.copyMakeBorder(
        image, pad_y, pad_y, pad_x, pad_x,
        borderType=cv2.BORDER_CONSTANT, value=(127, 127, 127),
    )


def generate_scut_embeddings(image_paths, filenames, cache_path, *,
                             model_name="buffalo_l", padding_fractions=(0.0, 0.25, 0.50),
                             face_app=None, image_reader=None):
    """Load cached SCUT ArcFace embeddings or generate them with padded retries."""
    paths, filenames = list(map(Path, image_paths)), np.asarray(filenames, dtype=str)
    cache_path = Path(cache_path).expanduser()
    if len(paths) != len(filenames) or len(set(filenames)) != len(filenames):
        raise ValueError("SCUT image paths and unique filenames must be aligned")
    if cache_path.is_file():
        return load_embeddings(cache_path)
    try:
        import cv2
    except ImportError as error:
        raise ImportError("Generating SCUT embeddings requires opencv-python") from error
    if face_app is None:
        try:
            from insightface.app import FaceAnalysis
        except ImportError as error:
            raise ImportError("Generating SCUT embeddings requires insightface") from error
        face_app = FaceAnalysis(
            name=model_name, allowed_modules=["detection", "recognition"],
            providers=["CPUExecutionProvider"],
        )
        face_app.prepare(ctx_id=-1, det_size=(640, 640))
    image_reader = cv2.imread if image_reader is None else image_reader
    embedded_names, vectors, failures = [], [], []
    padding_used = {fraction: 0 for fraction in padding_fractions}
    multiple_face_images = 0
    for path, filename in zip(paths, filenames):
        image = image_reader(str(path))
        if image is None:
            failures.append((filename, "image read failed"))
            continue
        faces, used_fraction = [], None
        for fraction in padding_fractions:
            faces = face_app.get(_neutral_padding(image, fraction, cv2))
            if faces:
                used_fraction = fraction
                break
        if not faces:
            failures.append((filename, f"no face detected after padding {tuple(padding_fractions)}"))
            continue
        padding_used[used_fraction] += 1
        multiple_face_images += int(len(faces) > 1)
        face = max(faces, key=lambda item: float(
            (item.bbox[2] - item.bbox[0]) * (item.bbox[3] - item.bbox[1])
        ))
        embedded_names.append(filename)
        vectors.append(np.asarray(face.normed_embedding, dtype=np.float32))
    if not vectors:
        raise ValueError("No faces were embedded")
    matrix = np.stack(vectors)
    if matrix.ndim != 2 or matrix.shape[1] != 512 or not np.isfinite(matrix).all():
        raise ValueError(f"Expected finite 512-dimensional ArcFace embeddings, got {matrix.shape}")
    identifiers = np.asarray(embedded_names)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, filenames=identifiers, embeddings=matrix)
    return {
        "matrix": matrix,
        "identifiers": identifiers,
        "source": str(cache_path),
        "diagnostics": {
            "padding_used": padding_used,
            "multiple_face_images": multiple_face_images,
            "failures": tuple(failures),
        },
    }
