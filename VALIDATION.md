# Validation record — 7 October 2026

This is a record of execution checks, not an accuracy evaluation or a claim that the app is already publicly deployed.

## Uploaded artifacts

| Check | Result |
| --- | --- |
| `embedding.pkl` | 8,664 float32 arrays, each `(2048,)` |
| Combined matrix | `(8664, 2048)`, finite values, no zero-norm rows |
| `filenames.pkl` | 8,664 unique paths; 100 celebrity folders; all paths under `data/` |
| Original scripts and both pickle files | Byte-for-byte equality with the uploaded copies confirmed |
| Full gallery / personal sample | Not provided |

SHA-256 of `embedding.pkl`:

`d1e81adfab1f966bea7723a5c2069706ad5e7920bf73f253e4098fc530aa3c72`

SHA-256 of `filenames.pkl`:

`932b3ce0dcab25b9702b39111fc51dfce8a52b51b3acfcd8d90ce7abd62d4a0c`

## Dependency and model execution

- Installed the requirements in a fresh **Linux x86_64 Python 3.11.16** environment.
- TensorFlow CPU 2.15.1, Keras 2.15.0, keras-vggface 0.6, MTCNN 0.1.0, NumPy 1.26.4, Pillow 9.5.0, OpenCV 4.8.1.78, Streamlit 1.41.1, and setuptools 75.8.0 were used.
- Dependency consistency check passed for 78 installed distributions.
- Binary-wheel-only resolution passed for Linux x86_64 / glibc 2.31 / Python 3.11 (78 distributions), matching the glibc generation of the documented Community Cloud Debian environment.
- Binary-wheel-only resolution passed for Windows x86_64 / Python 3.11 (79 distributions). This checks dependency availability and compatibility metadata; Windows execution was not performed here.
- The Windows pin for `tensorflow-io-gcs-filesystem` is 0.31.0 because 0.37.1 has no Windows wheel. Linux retains 0.37.1.
- MTCNN's `pkg_resources` import required setuptools 75.8.0; a newer automatically resolved setuptools release had removed it.
- The actual pretrained VGGFace ResNet50 weights downloaded successfully from the upstream release and loaded. Model name: `vggface_resnet50`; output shape: `(None, 2048)`.
- Ran a real image through detection, crop, preprocessing version 1, feature extraction, and cosine matching against the actual uploaded embeddings. The test used the package repository's [example image](https://github.com/rcmalli/keras-vggface/blob/master/image/ajb.jpg), which is not an uploaded gallery image. No expected celebrity label or accuracy claim was attached to this smoke test.

The upstream no-top ResNet50 weights file was 94,694,792 bytes. Its SHA-256 was:

`ad81577b744b2c8cc60cfb953747670cef18c3eed56ffff6007a00088614b14f`

The weights and upstream example image are not bundled in the deliverable. Weights are downloaded on first model initialization unless the original matching file is supplied in `models/`.

## Application checks

Eight focused unittest checks passed:

1. Invalid, empty, and oversized upload rejection.
2. Grayscale/RGBA image decoding to RGB.
3. Partial bounding-box clipping and rejection of empty/outside boxes.
4. No-face and multiple-face guards without calling the feature extractor; detector responses were stubbed for this test.
5. Celebrity-name extraction from a Windows-style filename.
6. Vectorized matching versus a scalar cosine formula on deterministic test vectors.
7. Self-comparison of selected real uploaded gallery vectors.
8. Thumbnail creation, path preservation, and source-image preservation using a synthetic temporary fixture.

Streamlit's `AppTest` also passed these application-level checks. The upload widget's return value was supplied by the test harness; browser file-transfer mechanics were not exercised:

- Initial page renders with the matching button disabled until a valid image is supplied.
- An actual face photo produces a displayed name and `Similarity Score` through the real models.
- A missing gallery thumbnail produces an informational message without hiding the name/score.
- A real blank image produces the no-face message through MTCNN.
- Changing the uploaded image clears the previous result.
- Invalid bytes named `.png` are rejected and disable the action.

Peak resident memory observed in that test process was approximately **837 MiB**. This is an observation for that local test, not a cloud memory guarantee, latency benchmark, or concurrency guarantee. Larger images, server overhead, and other sessions can increase usage.

All project Python files parsed successfully. `prepare_gallery.py` also parsed under Python 3.7 grammar, and it uses Pillow APIs available in the user's reported Pillow 9.5.0. Its full execution on the user's original Windows environment is still pending.

## Remaining checks before calling the demo ready

- Generate the actual thumbnails on the user's PC. None were available here, so the full matching-image collection has not been verified.
- Run `verify_model.py --reference-index 0 --source-root ...` against an original full-size gallery image. This is needed to compare new-runtime output with an existing stored vector.
- Compare an actual user sample with the original `test.py`. Exact equivalence to TensorFlow 2.3.1 was not measured.
- Confirm the intended channel-order choice. Default `legacy_bgr` preserves the original inference code; `rgb` is an explicit behavior change.
- Deploy to the user's GitHub/Streamlit accounts, test the public URL and actual host memory/cold start, then add the portfolio link.
- Docker build, Windows execution, and a publicly hosted app were not tested here.

There is no measured classification accuracy, calibrated confidence, or verification of which preprocessing history originally produced every row. Reference count and dimensions are verified; broader claims should not be inferred from them.

