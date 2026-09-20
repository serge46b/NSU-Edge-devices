#!/usr/bin/env python3
import argparse
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tmpwebrtc import imshow, start, stop  # noqa: E402

HERE = Path(__file__).resolve().parent
CHECK_JPG = HERE / "demo_recv.jpg"


def synthetic_frame(t):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (32, 32, 32)
    x = int((t * 120) % 600)
    cv2.rectangle(frame, (x, 200), (x + 40, 280), (0, 200, 255), -1)
    cv2.putText(
        frame,
        f"{t:.1f}s",
        (20, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.4,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return frame


def check_frame():
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:] = (180, 60, 20)
    cv2.rectangle(frame, (80, 80), (560, 400), (0, 255, 255), 8)
    cv2.putText(
        frame,
        "tmpwebrtc",
        (120, 260),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.6,
        (255, 255, 255),
        3,
        cv2.LINE_AA,
    )
    return frame


def run_loop():
    start()
    t0 = time.time()
    try:
        while True:
            imshow(synthetic_frame(time.time() - t0))
            time.sleep(1 / 30)
    except KeyboardInterrupt:
        pass
    finally:
        stop()


def run_check(host="127.0.0.1", port=8080):
    start(host="0.0.0.0", port=port)
    imshow(check_frame())
    time.sleep(0.2)
    base = f"http://{host}:{port}"
    try:
        with urllib.request.urlopen(base + "/", timeout=5) as r:
            html = r.read().decode("utf-8", errors="replace")
        if "/frame.jpg" not in html:
            raise RuntimeError("GET / did not return the stream page")
        print(f"GET / ok ({len(html)} bytes)")
        with urllib.request.urlopen(base + "/frame.jpg", timeout=5) as r:
            data = r.read()
        CHECK_JPG.write_bytes(data)
        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError("frame.jpg is not a JPEG")
        print(f"saved {CHECK_JPG} {img.shape[1]}x{img.shape[0]}")
    finally:
        stop()


def main():
    parser = argparse.ArgumentParser(description="tmpwebrtc demo")
    parser.add_argument(
        "--check",
        action="store_true",
        help="headless: GET / and save one JPEG frame",
    )
    args = parser.parse_args()
    if args.check:
        run_check()
    else:
        run_loop()


if __name__ == "__main__":
    main()
