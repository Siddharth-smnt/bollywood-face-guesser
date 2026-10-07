"""The uploaded project's VGGFace/MTCNN/cosine pipeline, separate from UI."""

import io
import os
import pickle
import threading
import warnings
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD_BYTES = 8 * 1024 * 1024
MAX_PIXELS = 12_000_000
MAX_DETECTION_SIDE = 1280
PREPROCESS_VERSION = 1  # Matches the supplied generator's upstream default.
COLOR_MODES = ("legacy_bgr", "rgb")


class InputError(ValueError):
    """A recoverable problem that can be explained to the visitor."""


class ArtifactError(RuntimeError):
    """Missing or inconsistent deployment assets."""


class _ArrayUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {
            ("numpy.core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
            ("numpy._core.multiarray", "_reconstruct"): np.core.multiarray._reconstruct,
            ("numpy", "ndarray"): np.ndarray,
            ("numpy", "dtype"): np.dtype,
        }
        try:
            return allowed[(module, name)]
        except KeyError:
            raise pickle.UnpicklingError("Unexpected object in project artifact")


def read_pickle(path):
    with Path(path).open("rb") as handle:
        return _ArrayUnpickler(handle).load()


def normalized_path(value):
    if not isinstance(value, str):
        raise ArtifactError("Every filename must be text.")
    path = PurePosixPath(value.replace("\\", "/"))
    if (path.is_absolute() or ".." in path.parts or len(path.parts) != 3
            or path.parts[0] != "data" or any(":" in p for p in path.parts)):
        raise ArtifactError("Expected data/Celebrity_Name/image.jpg paths.")
    return path


def celebrity_name(filename):
    return normalized_path(filename).parent.name.replace("_", " ")


@dataclass
class Gallery:
    filenames: tuple
    unit_embeddings: np.ndarray

    def closest(self, embedding):
        query = np.asarray(embedding, dtype=np.float32).reshape(-1)
        if query.shape != (2048,) or not np.isfinite(query).all():
            raise ArtifactError("The model returned an invalid embedding.")
        length = np.linalg.norm(query)
        if length <= 0:
            raise InputError("This photo could not be compared. Try a clearer face photo.")
        # Same cosine similarity as the original loop, computed in one operation.
        similarities = self.unit_embeddings @ (query / length)
        index = int(np.argmax(similarities))  # First entry wins a tie, as before.
        return index, float(np.clip(similarities[index], -1.0, 1.0))


def load_gallery(root):
    root = Path(root)
    try:
        filenames = read_pickle(root / "filenames.pkl")
        rows = read_pickle(root / "embedding.pkl")
        if not isinstance(filenames, (list, tuple)) or not filenames:
            raise ArtifactError("filenames.pkl must contain a nonempty list.")
        for filename in filenames:
            normalized_path(filename)
        features = np.asarray(rows, dtype=np.float32)
        del rows
        if features.shape != (len(filenames), 2048):
            raise ArtifactError("Embeddings and filenames must have matching rows and 2048 features.")
        if not np.isfinite(features).all():
            raise ArtifactError("The embedding file contains invalid numbers.")
        lengths = np.linalg.norm(features, axis=1, keepdims=True)
        if np.any(lengths <= 0):
            raise ArtifactError("The embedding file contains a zero-length vector.")
        features /= lengths
        features.setflags(write=False)
        return Gallery(tuple(filenames), features)
    except ArtifactError:
        raise
    except (OSError, ValueError, TypeError, pickle.UnpicklingError, EOFError) as exc:
        raise ArtifactError("Could not load embedding.pkl and filenames.pkl.") from exc


def decode_image(content):
    if not content or len(content) > MAX_UPLOAD_BYTES:
        raise InputError("Choose a JPG, PNG, or WebP image smaller than 8 MB.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as probe:
                if probe.format not in {"JPEG", "PNG", "WEBP"}:
                    raise InputError("Please use a JPG, PNG, or WebP image.")
                if probe.width * probe.height > MAX_PIXELS:
                    raise InputError("This image is too large. Please use a photo under 12 megapixels.")
                if getattr(probe, "n_frames", 1) != 1:
                    raise InputError("Please upload a still photo rather than an animation.")
                probe.verify()
            with Image.open(io.BytesIO(content)) as source:
                return ImageOps.exif_transpose(source).convert("RGB")
    except InputError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, SyntaxError,
            Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise InputError("That file could not be read as an image. Please choose another photo.") from exc


def clip_box(box, width, height):
    if len(box) != 4:
        raise InputError("The detected face could not be cropped. Try another photo.")
    x, y, w, h = map(int, box)
    left, top = max(0, x), max(0, y)
    right, bottom = min(width, x + w), min(height, y + h)
    if w <= 0 or h <= 0 or left >= right or top >= bottom:
        raise InputError("The detected face is outside the photo. Try another photo.")
    return left, top, right, bottom


@dataclass
class Match:
    index: int
    name: str
    filename: str
    similarity: float
    face: Image.Image
    resized_for_detection: bool


class FaceGuesser:
    def __init__(self, root, color_mode="legacy_bgr"):
        if color_mode not in COLOR_MODES:
            raise ArtifactError("FACE_COLOR_MODE must be legacy_bgr or rgb.")
        os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
        os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "2")
        os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
        os.environ.setdefault("OMP_NUM_THREADS", "2")
        os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

        from vggface_compat import enable_legacy_imports
        enable_legacy_imports()
        from keras_vggface.vggface import VGGFace
        from keras_vggface.utils import preprocess_input
        from mtcnn import MTCNN

        self.root = Path(root)
        self.gallery = load_gallery(root)
        self.color_mode = color_mode
        self.preprocess_input = preprocess_input
        local_weights = self.root / "models" / "rcmalli_vggface_tf_notop_resnet50.h5"
        self.model = VGGFace(
            model="resnet50", include_top=False, input_shape=(224, 224, 3),
            pooling="avg", weights=None if local_weights.is_file() else "vggface",
        )
        if local_weights.is_file():
            self.model.load_weights(str(local_weights))
        self.detector = MTCNN()
        self.lock = threading.RLock()  # Cached models are shared between sessions.
        if self.model.output_shape != (None, 2048):
            raise ArtifactError("Unexpected VGGFace output shape.")

    def guess(self, rgb_image):
        image = rgb_image.copy()
        original_size = image.size
        image.thumbnail((MAX_DETECTION_SIDE, MAX_DETECTION_SIDE), Image.Resampling.LANCZOS)
        rgb_array = np.asarray(image)
        # Deliberately retain the uploaded test.py channel order by default.
        pixels = rgb_array[..., ::-1].copy() if self.color_mode == "legacy_bgr" else rgb_array
        with self.lock:
            faces = self.detector.detect_faces(pixels)
            if not faces:
                raise InputError("No face found. Try a clear, well-lit photo looking toward the camera.")
            if len(faces) > 1:
                raise InputError("We found more than one face. Please crop your photo to one person and try again.")
            left, top, right, bottom = clip_box(faces[0]["box"], image.width, image.height)
            face = Image.fromarray(pixels[top:bottom, left:right]).resize(
                (224, 224), Image.Resampling.BICUBIC
            )
            batch = np.expand_dims(np.asarray(face, dtype=np.float32), axis=0)
            batch = self.preprocess_input(batch, version=PREPROCESS_VERSION)
            embedding = self.model.predict(batch, verbose=0).flatten()
        index, score = self.gallery.closest(embedding)
        filename = self.gallery.filenames[index]
        return Match(index, celebrity_name(filename), filename, score,
                     image.crop((left, top, right, bottom)), image.size != original_size)

    def matched_image_path(self, filename):
        path = normalized_path(filename)
        # Thumbnails are display-only, preserving the matched reference filename.
        for base in (self.root / "gallery", self.root):
            candidate = base.joinpath(*path.parts)
            if candidate.is_file():
                return candidate
        return None

