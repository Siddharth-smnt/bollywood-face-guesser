"""Focused checks for upload handling, boundaries, matching, and thumbnail safety.

Run: python -m unittest test_app_helpers -v
These tests do not download model weights or use a real face photo.
"""

import io
import pickle
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import Mock

import numpy as np
from PIL import Image

from inference import (FaceGuesser, Gallery, InputError, celebrity_name,
                       clip_box, decode_image, load_gallery)
from prepare_gallery import build_gallery


class InputTests(unittest.TestCase):
    def test_invalid_and_oversized_uploads(self):
        for content in (b"", b"not an image", b"x" * (8 * 1024 * 1024 + 1)):
            with self.assertRaises(InputError):
                decode_image(content)

    def test_rgba_and_gray_images_become_rgb(self):
        for mode in ("RGBA", "L"):
            stream = io.BytesIO()
            Image.new(mode, (16, 20)).save(stream, "PNG")
            result = decode_image(stream.getvalue())
            self.assertEqual(result.mode, "RGB")
            self.assertEqual(result.size, (16, 20))

    def test_partial_and_invalid_boxes(self):
        self.assertEqual(clip_box([-10, -5, 30, 20], 100, 100), (0, 0, 20, 15))
        self.assertEqual(clip_box([90, 90, 30, 20], 100, 100), (90, 90, 100, 100))
        for box in ([120, 20, 10, 10], [4, 3, -2, 10], [-20, 5, 10, 10]):
            with self.assertRaises(InputError):
                clip_box(box, 100, 100)

    def test_no_face_and_multiple_faces_do_not_call_model(self):
        engine = object.__new__(FaceGuesser)
        engine.color_mode = "legacy_bgr"
        engine.lock = threading.RLock()
        engine.detector = Mock()
        engine.model = Mock()
        for boxes, expected in (([], "No face found"),
                                ([{"box": [1, 1, 5, 5]}] * 2, "more than one face")):
            engine.detector.detect_faces.return_value = boxes
            with self.assertRaisesRegex(InputError, expected):
                engine.guess(Image.new("RGB", (20, 20)))
        engine.model.predict.assert_not_called()

    def test_celebrity_from_windows_filename(self):
        self.assertEqual(celebrity_name(r"data\Aamir_Khan\Aamir.40.jpg"), "Aamir Khan")

    def test_matching_agrees_with_scalar_cosine(self):
        rng = np.random.default_rng(42)
        vectors = rng.normal(size=(32, 2048)).astype(np.float32)
        query = rng.normal(size=2048).astype(np.float32)
        unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
        gallery = Gallery(tuple(str(i) for i in range(32)), unit)
        expected = [np.dot(v, query) / (np.linalg.norm(v) * np.linalg.norm(query)) for v in vectors]
        index, score = gallery.closest(query)
        self.assertEqual(index, int(np.argmax(expected)))
        self.assertAlmostEqual(score, float(max(expected)), places=6)

    def test_uploaded_database_self_matches(self):
        gallery = load_gallery(Path(__file__).resolve().parent)
        self.assertEqual(gallery.unit_embeddings.shape, (8664, 2048))
        for index in (0, 4000, 8663):
            _, score = gallery.closest(gallery.unit_embeddings[index])
            self.assertAlmostEqual(score, 1.0, places=5)

    def test_thumbnails_keep_mapping_and_leave_source_intact(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = root / "original"
            source = original / "data" / "Aamir_Khan" / "Aamir.40.jpg"
            source.parent.mkdir(parents=True)
            Image.new("RGB", (1000, 500), "blue").save(source)
            original_bytes = source.read_bytes()
            with (original / "filenames.pkl").open("wb") as handle:
                pickle.dump([r"data\Aamir_Khan\Aamir.40.jpg"], handle)
            output = root / "web" / "gallery"
            report = build_gallery(original, output)
            self.assertEqual(report["written"], 1)
            self.assertFalse(report["errors"])
            self.assertEqual(source.read_bytes(), original_bytes)
            with Image.open(output / "data" / "Aamir_Khan" / "Aamir.40.jpg") as thumbnail:
                self.assertEqual(thumbnail.size, (256, 128))


if __name__ == "__main__":
    unittest.main()

