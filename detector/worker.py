from pathlib import Path
from queue import Queue, Empty
from threading import Event
import cv2.typing as cv2t
import cv2
import numpy as np
from rknnlite.api import RKNNLite


INPUT_SIZE = 640
CONF_THRES = 0.4
TIMEOUT = 2

HERE = Path(__file__).resolve().parent
MODEL_WEIGHTS = f"{HERE}/yolov12n-face-rk3588-fp16.rknn"


class Detector:
    __yolo_model: RKNNLite | None = None
    __in_queue: Queue[cv2t.MatLike]
    __out_queue: Queue[cv2t.MatLike]
    __filter_height_percent: float

    def __init__(
        self,
        in_queue: Queue[cv2t.MatLike],
        out_queue: Queue[cv2t.MatLike],
        stop_event: Event,
        filter_height_percent: float,
    ):
        self.__in_queue = in_queue
        self.__out_queue = out_queue
        self.__stop_event = stop_event
        self.__yolo_model = self._load_model(MODEL_WEIGHTS)
        self.__filter_height_percent = filter_height_percent

    def __del__(self):
        if self.__yolo_model:
            self.__yolo_model.release()

    def run(self):
        print("Detector: Thread started")
        while not self.__stop_event.is_set():
            try:
                try:
                    frame = self.__in_queue.get(timeout=TIMEOUT)
                except Empty:
                    continue
                cropped = self._process_frame(frame)
                if cropped is not None:
                    self.__out_queue.put(cropped)
            except Exception as e:
                print(f"Detector: Error processing frame: {e}")
                continue
        print("Detector: Thread stopped")

    def _load_model(self, model_weights: str):
        rknn = RKNNLite(verbose=False)
        if rknn.load_rknn(str(model_weights)) != 0:
            raise RuntimeError(f"Detector: load_rknn failed: {model_weights}")
        if rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_AUTO) != 0:
            raise RuntimeError(f"Detector: init_runtime failed: {model_weights}")
        return rknn

    def _process_frame(self, frame: cv2t.MatLike):
        if not self.__yolo_model:
            raise RuntimeError("Detector: Yolo model not loaded")
        h0, w0 = frame.shape[:2]
        model_bgr = cv2.resize(frame, (INPUT_SIZE, INPUT_SIZE))
        rgb = cv2.cvtColor(model_bgr, cv2.COLOR_BGR2RGB)
        inp = np.expand_dims(rgb, 0)
        pred = self.__yolo_model.inference(inputs=[inp])[0]
        boxes, scores = self._decode_yolo(pred, conf_thres=CONF_THRES)
        boxes_full = self._scale_boxes(boxes.copy(), w0, h0)

        if len(scores) == 0:
            return None
        best_idx = np.argmax(scores)
        best_box = boxes_full[best_idx]

        x1, y1, x2, y2 = best_box.astype(int)
        if abs(y2 - y1) < h0 * self.__filter_height_percent:
            return None
        x1 = max(0, min(x1, w0 - 1))
        y1 = max(0, min(y1, h0 - 1))
        x2 = max(0, min(x2, w0 - 1))
        y2 = max(0, min(y2, h0 - 1))

        cropped = frame[y1:y2, x1:x2]
        return cropped

    def _scale_boxes(self, boxes, w0, h0):
        if len(boxes) == 0:
            return boxes
        out = boxes.copy()
        out[:, [0, 2]] *= w0 / INPUT_SIZE
        out[:, [1, 3]] *= h0 / INPUT_SIZE
        return out

    def _decode_yolo(self, pred, conf_thres=0.4):
        p = np.asarray(pred).reshape(5, -1)
        scores = p[4]
        keep = scores >= conf_thres
        if not np.any(keep):
            return np.zeros((0, 4)), np.zeros((0,))
        boxes = p[:4, keep].T.copy()
        scores = scores[keep]
        idx = self._nms(boxes, scores)
        return boxes[idx], scores[idx]

    def _nms(self, boxes, scores, iou_thres=0.45):
        order = scores.argsort()[::-1]
        keep = []
        while order.size:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
            area_i = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1])
            area_r = (boxes[rest, 2] - boxes[rest, 0]) * (
                boxes[rest, 3] - boxes[rest, 1]
            )
            iou = inter / (area_i + area_r - inter + 1e-6)
            order = rest[iou < iou_thres]
        return keep
