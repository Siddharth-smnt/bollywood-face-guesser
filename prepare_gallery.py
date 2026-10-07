"""Create small display copies of the EXACT images named in filenames.pkl.

Works with Python 3.7 and Pillow 9.5 already installed in the original project.
Does not run ML, regenerate embeddings, or modify source images.
"""

import argparse
import json
import pickle
from pathlib import Path, PurePosixPath

from PIL import Image, ImageOps


class FilenamesUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError("Expected only a list of filename strings")


def build_gallery(source_root, output_root, max_side=256, quality=75):
    source_root, output_root = Path(source_root).resolve(), Path(output_root).resolve()
    original_data = source_root / "data"
    if output_root == source_root or output_root == original_data or original_data in output_root.parents:
        raise ValueError("Choose a separate output folder, such as the web app's gallery folder.")
    with (source_root / "filenames.pkl").open("rb") as handle:
        filenames = FilenamesUnpickler(handle).load()
    if not isinstance(filenames, (list, tuple)) or not filenames:
        raise ValueError("filenames.pkl must be a nonempty list")
    if max_side < 64 or max_side > 1024 or not 20 <= quality <= 95:
        raise ValueError("Use max-side 64–1024 and JPEG quality 20–95")
    report = {"expected": len(filenames), "written": 0, "bytes": 0, "errors": []}
    for number, filename in enumerate(filenames, 1):
        try:
            if not isinstance(filename, str):
                raise ValueError("Invalid filename")
            relative = PurePosixPath(filename.replace("\\", "/"))
            if (len(relative.parts) != 3 or relative.parts[0] != "data"
                    or relative.is_absolute() or ".." in relative.parts
                    or any(":" in part for part in relative.parts)
                    or relative.suffix.lower() not in {".jpg", ".jpeg"}):
                raise ValueError("Expected data/Celebrity_Name/image.jpg")
            source = source_root.joinpath(*relative.parts)
            target = output_root.joinpath(*relative.parts)
            if source.resolve() == target.resolve():
                raise ValueError("Refusing to overwrite an original image")
            with Image.open(source) as image:
                thumb = ImageOps.exif_transpose(image).convert("RGB")
                thumb.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
                target.parent.mkdir(parents=True, exist_ok=True)
                thumb.save(target, "JPEG", quality=quality, optimize=True)
            report["written"] += 1
            report["bytes"] += target.stat().st_size
        except (OSError, ValueError, TypeError, Image.DecompressionBombError) as exc:
            report["errors"].append({"filename": str(filename), "error": str(exc)})
        if number % 500 == 0:
            print("Processed {}/{} images".format(number, len(filenames)), flush=True)
    output_root.mkdir(parents=True, exist_ok=True)
    report_path = output_root.parent / "gallery_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Created {} of {} thumbnails ({:.1f} MiB).".format(
        report["written"], report["expected"], report["bytes"] / 1024**2))
    print("Report: {}".format(report_path))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, help="Original project folder containing data/ and filenames.pkl")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent / "gallery"))
    parser.add_argument("--max-side", type=int, default=256)
    parser.add_argument("--quality", type=int, default=75)
    args = parser.parse_args()
    try:
        report = build_gallery(args.source_root, args.output, args.max_side, args.quality)
    except (OSError, ValueError, pickle.UnpicklingError) as exc:
        parser.exit(1, "Gallery preparation failed: {}\n".format(exc))
    if report["errors"]:
        parser.exit(2, "Some images are missing or unreadable; see gallery_report.json. Matching still works, but those previews will be unavailable.\n")


if __name__ == "__main__":
    main()

