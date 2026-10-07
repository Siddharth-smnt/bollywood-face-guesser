# Bollywood Celebrity Face Guesser

A Streamlit application that finds a Bollywood celebrity lookalike from an uploaded photo. It uses **MTCNN** for face detection, a **pretrained VGGFace ResNet50** network for feature extraction, and **cosine similarity** to find the closest reference image.

The network is used as a feature extractor; this project does not train a celebrity classifier from scratch.

## How it works

1. Validate the uploaded image and detect a face with MTCNN.
2. Crop the face, clip the bounding box to the image, and resize the crop to **224 × 224**.
3. Apply VGGFace preprocessing and extract a **2,048-dimensional embedding** using ResNet50 with average pooling and no classification head.
4. Compare the embedding with the stored celebrity embeddings using cosine similarity.
5. Display the nearest reference image, the celebrity name derived from its folder, and a similarity score.

**Similarity Score** is cosine similarity multiplied by 100. It is not a probability, recognition accuracy, or calibrated confidence. The app always returns the closest available gallery entry; it does not verify identity.

## Features

- JPG, PNG, and WebP photo uploads.
- Side-by-side input and matching celebrity images, with a face-crop preview.
- Helpful messages for no face, multiple faces, invalid files, and oversized images.
- EXIF orientation correction and bounded face crops.
- Cached model loading and vectorized similarity comparison.
- Separate inference logic and Streamlit interface.

## Run locally

Use **64-bit Python 3.11**. Dependency versions are pinned in `requirements.txt` and `constraints.txt`.

### Windows PowerShell

```powershell
git clone https://github.com/Siddharth-smnt/bollywood-face-guesser.git
cd bollywood-face-guesser
py -3.11 -m venv .venv-web
.\.venv-web\Scripts\python.exe -m pip install -r requirements.txt
.\.venv-web\Scripts\python.exe -m streamlit run app.py
```

### Linux

```bash
git clone https://github.com/Siddharth-smnt/bollywood-face-guesser.git
cd bollywood-face-guesser
python3.11 -m venv .venv-web
.venv-web/bin/python -m pip install -r requirements.txt
.venv-web/bin/python -m streamlit run app.py
```

On Debian-based Linux systems, the OpenCV system dependencies are listed in `packages.txt`. Open the address printed by Streamlit, normally [http://localhost:8501](http://localhost:8501).

VGGFace weights download on the first model initialization. The loader also supports a local copy at `models/rcmalli_vggface_tf_notop_resnet50.h5`.

## Reference gallery and preprocessing

`embedding.pkl` and `filenames.pkl` form a paired reference database: each embedding corresponds to the filename at the same position. Paths use the structure `data/Celebrity_Name/image.jpg`. Matching-photo thumbnails are stored under `gallery/data/` and are used for display only.

The default `FACE_COLOR_MODE=legacy_bgr` preserves the original inference script's BGR channel order. The original embedding generator loads RGB images, so this channel-order difference remains a known compatibility limitation. Preprocessing uses `preprocess_input(..., version=1)` to match the original generator's default. An optional `FACE_COLOR_MODE=rgb` mode is available for comparison; it can change detections and matches.

The compatibility adapter in `vggface_compat.py` provides legacy import locations for `keras-vggface`. It does not replace the network layers or pretrained weights.

## Project structure

| Path | Purpose |
| --- | --- |
| `app.py` | Streamlit upload and results interface |
| `inference.py` | Image validation, face detection, feature extraction, and matching |
| `vggface_compat.py` | Keras import compatibility adapter |
| `embedding.pkl`, `filenames.pkl` | Stored embeddings and their reference-image mapping |
| `gallery/` | Display thumbnails |
| `prepare_gallery.py` | Thumbnail generation from source images |
| `verify_model.py` | Model startup and optional reference-embedding checks |
| `test_app_helpers.py` | Helper tests |
| `originals/` | Original feature-extraction and inference scripts |
| `requirements.txt`, `constraints.txt`, `packages.txt` | Python and Linux dependencies |

## Checks

From the project directory in Windows PowerShell:

```powershell
.\.venv-web\Scripts\python.exe -m pip check
.\.venv-web\Scripts\python.exe -m unittest test_app_helpers -v
.\.venv-web\Scripts\python.exe verify_model.py
```

On Linux, use `.venv-web/bin/python` for these commands. `verify_model.py` initializes the model and checks the no-face response on a blank image. Its optional reference comparison checks embedding reproduction, not classification accuracy.

## Technology

Python · Streamlit · TensorFlow/Keras · keras-vggface · MTCNN · NumPy · OpenCV · Pillow

Developed by [Siddharth Samant](https://github.com/Siddharth-smnt) · [Portfolio](https://www.whoissid.com)
