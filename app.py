"""Streamlit interface. Run: python -m streamlit run app.py"""

import hashlib
import logging
import os
from pathlib import Path

import streamlit as st
from PIL import Image

from inference import ArtifactError, FaceGuesser, InputError, decode_image

ROOT = Path(__file__).resolve().parent
LOGGER = logging.getLogger(__name__)

st.set_page_config(page_title="Bollywood Celebrity Guesser", page_icon="🎬", layout="centered")


@st.cache_resource(max_entries=1, show_spinner=False)
def get_engine(color_mode):
    return FaceGuesser(ROOT, color_mode=color_mode)


def main():
    st.title("🎬 Bollywood Celebrity Guesser")
    st.write("Upload a photo and discover your closest match in a Bollywood celebrity gallery.")
    uploaded = st.file_uploader(
        "Upload your photo", type=["jpg", "jpeg", "png", "webp"],
        help="Use a clear photo with one face. Maximum file size: 8 MB.",
    )
    color_mode = os.environ.get("FACE_COLOR_MODE", "legacy_bgr")
    photo = None
    upload_key = None
    if uploaded is not None:
        content = uploaded.getvalue()
        upload_key = (hashlib.sha256(content).hexdigest(), color_mode)
        try:
            photo = decode_image(content)
        except InputError as exc:
            st.error(str(exc))

    if st.session_state.get("upload_key") != upload_key:
        st.session_state.pop("match", None)
        st.session_state.pop("match_path", None)
        st.session_state["upload_key"] = upload_key

    if photo is not None and "match" not in st.session_state:
        st.image(photo, caption="Your photo", width=280)

    if st.button("Find My Lookalike", type="primary", disabled=photo is None, use_container_width=True):
        st.session_state.pop("match", None)
        st.session_state.pop("match_path", None)
        try:
            with st.spinner("Finding your closest match… The first run may take a little longer."):
                engine = get_engine(color_mode)
                match = engine.guess(photo)
                match_path = engine.matched_image_path(match.filename)
            st.session_state["match"] = match
            st.session_state["match_path"] = str(match_path) if match_path else None
            st.rerun()
        except InputError as exc:
            st.warning(str(exc))
        except ArtifactError:
            LOGGER.exception("Project asset validation failed")
            st.error("The reference gallery could not be loaded. Please try again later.")
        except Exception:
            LOGGER.exception("Face matching failed")
            st.error("The model could not finish loading or processing this photo. Please try again shortly.")

    match = st.session_state.get("match")
    if match is not None and photo is not None:
        st.divider()
        st.subheader("Your Result")
        left, right = st.columns(2)
        with left:
            st.image(photo, caption="Your photo", use_container_width=True)
        with right:
            match_path = st.session_state.get("match_path")
            if match_path:
                try:
                    with Image.open(match_path) as reference:
                        st.image(reference.convert("RGB"), caption=match.name, use_container_width=True)
                except (OSError, ValueError):
                    st.info("The matching photo is unavailable, but the name and score are shown below.")
            else:
                st.info("The matching photo has not been added to this demo yet.")
        st.write("You look most like:")
        st.subheader(match.name)
        st.metric("Similarity Score", "{:.1f}%".format(match.similarity * 100))
        st.caption("Cosine similarity × 100. This score is not a probability or confidence level.")
        with st.expander("See the face used for matching"):
            st.image(match.face, width=224)
            if match.resized_for_detection:
                st.caption("Your large photo was resized before face detection to keep the demo responsive.")

    with st.expander("How it works"):
        st.write(
            "MTCNN detects and crops a face. A pretrained VGGFace ResNet50 model "
            "turns the resized face into a feature embedding. The app compares it "
            "with stored celebrity embeddings using cosine similarity and returns "
            "the closest matching reference image."
        )
        st.caption("This is a lookalike demo. It always selects the closest available match and does not verify identity.")
    st.caption("Photos are processed in memory and are not saved by this app.")
    st.markdown("[← Back to Sid’s portfolio](https://www.whoissid.com)")


if __name__ == "__main__":
    main()

