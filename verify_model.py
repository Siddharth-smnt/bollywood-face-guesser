"""Load the real models and optionally compare a reference against its stored vector."""

import argparse
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

from inference import FaceGuesser, InputError, decode_image, normalized_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--photo", type=Path, help="Optional uploaded-style photo for a full inference run")
    parser.add_argument("--reference-index", type=int, help="Optional index in filenames.pkl to reproduce with the ORIGINAL generator preprocessing")
    parser.add_argument("--source-root", type=Path, help="Original project folder, needed for reference comparison")
    args = parser.parse_args()
    if args.reference_index is not None and args.source_root is None:
        parser.error("--reference-index requires --source-root")
    engine = FaceGuesser(Path(__file__).resolve().parent, os.environ.get("FACE_COLOR_MODE", "legacy_bgr"))
    report = {
        "model": engine.model.name,
        "model_output_shape": engine.model.output_shape,
        "gallery_shape": engine.gallery.unit_embeddings.shape,
        "color_mode": engine.color_mode,
        "preprocessing_version": 1,
    }
    if args.reference_index is not None:
        index = args.reference_index
        if not 0 <= index < len(engine.gallery.filenames):
            parser.error("reference-index is outside the gallery")
        filename = engine.gallery.filenames[index]
        source = args.source_root.joinpath(*normalized_path(filename).parts)
        # Reproduce feature_extractor.py: load_img defaults to RGB and NEAREST;
        # no MTCNN cropping here. Use original images, NOT display thumbnails.
        from tensorflow.keras.preprocessing import image as keras_image
        picture = keras_image.load_img(source, target_size=(224, 224))
        batch = np.expand_dims(keras_image.img_to_array(picture), axis=0)
        batch = engine.preprocess_input(batch, version=1)
        vector = engine.model.predict(batch, verbose=0).flatten()
        unit = vector / np.linalg.norm(vector)
        cosine = float(np.dot(unit, engine.gallery.unit_embeddings[index]))
        report["reference_comparison"] = {"index": index, "filename": filename, "cosine_with_stored_vector": cosine}
        if cosine < 0.9999:
            print(json.dumps(report, indent=2))
            raise SystemExit("Reference reproduction differs. Keep existing embeddings; investigate weights/preprocessing/source image before deployment.")
    if args.photo:
        try:
            match = engine.guess(decode_image(args.photo.read_bytes()))
            report["photo_result"] = {"name": match.name, "filename": match.filename, "cosine_similarity": match.similarity}
        except InputError as exc:
            report["photo_result"] = {"input_message": str(exc)}
    else:
        try:
            engine.guess(Image.new("RGB", (224, 224), "gray"))
        except InputError as exc:
            if "No face found" not in str(exc):
                raise
            report["blank_image_check"] = "No face found, as expected"
        else:
            raise SystemExit("Unexpected face detected in blank test image")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

