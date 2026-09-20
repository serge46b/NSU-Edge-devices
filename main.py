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

NUM_DETECTORS = 1
NUM_EMBEDDERS = 1
FILTER_HEIGHT_PERCENT = 0.5
COMPARISON_THRESHOLD = 0.5
CAMERA_DEVICE = "/dev/video20"
EMBEDDING_COLLECTION_TIMEOUT = 10

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
            Detector(frames_queue, faces_queue, stop_event, FILTER_HEIGHT_PERCENT)
        )
        detector_threads.append(
            Thread(target=detectors[i].run, args=(frames_queue, faces_queue))
        )
    embedders = []
    embedder_threads = []
    for i in range(NUM_EMBEDDERS):
        embedders.append(Embedder(faces_queue, embeddings_queue, stop_event))
        embedder_threads.append(
            Thread(target=embedders[i].run, args=(faces_queue, embeddings_queue))
        )
    opener = Opener(device_response_event, stop_event)
    comparator = Comparator(COMPARISON_THRESHOLD)
    cap = cv2.VideoCapture(CAMERA_DEVICE)
    if not cap.isOpened():
        raise RuntimeError(f"Main: Failed to open camera {CAMERA_DEVICE}")
    start_detection_time = None
    collected_embeddings = []
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                print("Main: Failed to read frame")
                stop_event.set()
                break
            if device_response_event.is_set():
                continue
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
            opener.run_command("PENDING")
            start_detection_time = monotonic()
            collected_embeddings.append(embedding)
            if len(collected_embeddings) < 5:
                continue
            while not frames_queue.empty():
                data = frames_queue.get_nowait()
            while not faces_queue.empty():
                data = faces_queue.get_nowait()
            while not embeddings_queue.empty():
                data = embeddings_queue.get_nowait()
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
        cap.release()
        for thread in detector_threads + embedder_threads:
            thread.join()
    print("Main: Stopped")
