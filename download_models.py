"""Download the YuNet and SFace models published by the OpenCV Zoo."""

from __future__ import annotations

import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODELS = {
    "face_detection_yunet_2023mar.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx",
        100_000,
    ),
    "face_recognition_sface_2021dec.onnx": (
        "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/models/"
        "face_recognition_sface/face_recognition_sface_2021dec.onnx",
        1_000_000,
    ),
}


def download(filename: str, url: str, minimum_size: int) -> None:
    destination = MODEL_DIR / filename
    if destination.exists() and destination.stat().st_size >= minimum_size:
        print(f"Already present: {filename}")
        return
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "ClassroomAttendanceSetup/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open("wb") as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if temporary.stat().st_size < minimum_size:
            raise RuntimeError("The downloaded file is unexpectedly small; it may be a Git LFS pointer.")
        with temporary.open("rb") as model_file:
            if model_file.read(40).startswith(b"version https://git-lfs"):
                raise RuntimeError("Git returned a Git LFS pointer instead of the model file.")
        temporary.replace(destination)
        print(f"Downloaded: {filename}")
    finally:
        temporary.unlink(missing_ok=True)


if __name__ == "__main__":
    for name, (source, minimum_size) in MODELS.items():
        download(name, source, minimum_size)
    print(f"Model files are ready in {MODEL_DIR}")
