from classifier.comparator import Comparator
from classifier.embedder import Embedder
from detector.worker import Detector
from opener.worker import Opener
from threading import Thread, Event, Condition
from queue import Queue, Full, Empty
import cv2.typing as cv2t
import cv2
import numpy as np
from time import monotonic
from tmpwebrtc import imshow, start, stop

NUM_DETECTORS = 1
NUM_EMBEDDERS = 1
FILTER_HEIGHT_PERCENT = 0.5
COMPARISON_THRESHOLD = 0.5
CAMERA_DEVICE = "/dev/video20"
EMBEDDING_COLLECTION_TIMEOUT = 10
TIME_BETWEEN_FRAMES = 0.1
EMBEDDINGS_AMOUNT = 5

if __name__ == "__main__":
    print("Main: Starting...")
    stop_event = Event()
    device_response_event = Event()
    frames_queue = Queue(maxsize=10)
    faces_queue = Queue(maxsize=5)
    embeddings_queue = Queue(maxsize=5)
    detectors = []
    detector_threads = []
    for i in range(NUM_DETECTORS):
        detectors.append(
            Detector(frames_queue, faces_queue, stop_event, device_response_event, FILTER_HEIGHT_PERCENT)
        )
        thread = Thread(target=detectors[i].run)
        thread.start()
        detector_threads.append(thread)
    embedders = []
    embedder_threads = []
    for i in range(NUM_EMBEDDERS):
        embedders.append(Embedder(faces_queue, embeddings_queue, stop_event, device_response_event))
        thread = Thread(target=embedders[i].run)
        thread.start()
        embedder_threads.append(thread)
    opener = Opener(device_response_event, stop_event)
    comparator = Comparator(COMPARISON_THRESHOLD)
    cap = cv2.VideoCapture(CAMERA_DEVICE)
    if not cap.isOpened():
        raise RuntimeError(f"Main: Failed to open camera {CAMERA_DEVICE}")
    start_detection_time = None
    collected_embeddings = []
    last_frame_time = monotonic()
    try:
        start()
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Main: Failed to read frame")
                stop_event.set()
                break
            if device_response_event.is_set() or monotonic() - last_frame_time < TIME_BETWEEN_FRAMES:
                continue
            imshow(frame)
            last_frame_time = monotonic()
            try:
                frames_queue.put_nowait(frame)
            except Full:
                frames_queue.get_nowait()
                frames_queue.put_nowait(frame)
            try:
                embedding = embeddings_queue.get_nowait()
            except Empty:
                if (
                    start_detection_time is not None
                    and monotonic() - start_detection_time
                    > EMBEDDING_COLLECTION_TIMEOUT
                ):
                    start_detection_time = None
                    collected_embeddings = []
                    opener.run_command("NOT_RECOGNIZED")
                continue
            if len(collected_embeddings) == 0:
                opener.run_command("PENDING")
            start_detection_time = monotonic()
            collected_embeddings.append(embedding)
            if len(collected_embeddings) < EMBEDDINGS_AMOUNT:
                continue
            while not frames_queue.empty():
                _ = frames_queue.get_nowait()
            while not faces_queue.empty():
                _ = faces_queue.get_nowait()
            while not embeddings_queue.empty():
                _ = embeddings_queue.get_nowait()
            name = comparator.find_person(collected_embeddings)
            start_detection_time = None
            collected_embeddings = []
            if name:
                opener.run_command("OK")
            else:
                opener.run_command("NOT_RECOGNIZED")
    except KeyboardInterrupt:
        stop_event.set()
    finally:
        stop()
        cap.release()
        for thread in detector_threads + embedder_threads:
            thread.join()
        print("Main: Stopped")
