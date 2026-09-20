from pathlib import Path
from queue import Queue, Empty
from threading import Event
import cv2.typing as cv2t
import cv2
import numpy as np
from rknnlite.api import RKNNLite

INPUT_SIZE = 112
TIMEOUT = 2

HERE = Path(__file__).resolve().parent
MODEL_WEIGHTS = f"{HERE}/w600k_r50-rk3588-fp16.rknn"


class Embedder:
    __model: RKNNLite
    __in_queue: Queue[cv2t.MatLike]
    __out_queue: Queue[np.ndarray]
    __stop_event: Event

    def __init__(
        self,
        in_queue: Queue[cv2t.MatLike],
        out_queue: Queue[np.ndarray],
        stop_event: Event,
    ):
        self.__in_queue = in_queue
        self.__out_queue = out_queue
        self.__stop_event = stop_event
        self.__model = self._load_rknn(MODEL_WEIGHTS)

    def __del__(self):
        if self.__model:
            self.__model.release()

    def run(self):
        print("Embedder: Thread started")
        while not self.__stop_event.is_set():
            try:
                frame = self.__in_queue.get(timeout=TIMEOUT)
            except Empty:
                continue
            embedding = self._process(frame)
            self.__out_queue.put(embedding)
        print("Embedder: Thread stopped")

    def _load_rknn(self, model_weights: str):
        rknn = RKNNLite(verbose=False)
        if rknn.load_rknn(str(model_weights)) != 0:
            raise RuntimeError(f"Embedder: load_rknn failed: {model_weights}")
        if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
            raise RuntimeError(f"Embedder: init_runtime failed: {model_weights}")
        return rknn

    def _preprocess(self, img: cv2t.MatLike):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (INPUT_SIZE, INPUT_SIZE))
        return np.expand_dims(rgb, 0)

    def _process(self, img: cv2t.MatLike):
        if not self.__model:
            raise RuntimeError("Embedder: Model not loaded")
        rgb = self._preprocess(img)
        pred = self.__model.inference(inputs=[rgb])[0]

        norms = np.linalg.norm(pred, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return pred / norms
