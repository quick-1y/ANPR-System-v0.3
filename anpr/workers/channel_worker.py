import asyncio
from datetime import datetime, timezone
from typing import Dict, Optional, Tuple

import cv2
from PyQt5 import QtCore, QtGui

from anpr.detection.motion_detector import MotionDetector, MotionDetectorConfig
from anpr.pipeline.factory import build_components
from logging_manager import get_logger
from storage import AsyncEventDatabase

logger = get_logger(__name__)


class InferenceLimiter:
    """Пропускает лишние кадры для инференса детектора."""

    def __init__(self, stride: int) -> None:
        self.stride = max(1, stride)
        self._counter = 0

    def allow(self) -> bool:
        should_run = self._counter == 0
        self._counter = (self._counter + 1) % self.stride
        return should_run


class ChannelWorker(QtCore.QThread):
    """Background worker that captures frames, runs ANPR pipeline and emits UI events."""

    frame_ready = QtCore.pyqtSignal(str, QtGui.QImage)
    event_ready = QtCore.pyqtSignal(dict)
    status_ready = QtCore.pyqtSignal(str, str)

    def __init__(self, channel_conf: Dict, db_path: str, parent=None) -> None:
        super().__init__(parent)
        self.channel_conf = channel_conf
        self.db_path = db_path
        self._running = True
        self.best_shots = int(channel_conf.get("best_shots", 3))
        self.cooldown_seconds = int(channel_conf.get("cooldown_seconds", 5))
        self.min_confidence = float(channel_conf.get("ocr_min_confidence", 0.6))
        self.detector_frame_stride = max(1, int(channel_conf.get("detector_frame_stride", 2)))
        self.detection_mode = channel_conf.get("detection_mode", "continuous")
        self.motion_threshold = float(channel_conf.get("motion_threshold", 0.01))
        self.motion_detector = MotionDetector(
            MotionDetectorConfig(
                threshold=self.motion_threshold,
                frame_stride=int(channel_conf.get("motion_frame_stride", 1)),
                activation_frames=int(channel_conf.get("motion_activation_frames", 3)),
                release_frames=int(channel_conf.get("motion_release_frames", 6)),
            )
        )
        self._inference_limiter = InferenceLimiter(self.detector_frame_stride)

    def _open_capture(self, source: str) -> Optional[cv2.VideoCapture]:
        capture = cv2.VideoCapture(int(source) if source.isnumeric() else source)
        if not capture.isOpened():
            return None
        return capture

    def _build_pipeline(self) -> Tuple[object, object]:
        return build_components(self.best_shots, self.cooldown_seconds, self.min_confidence)

    def _region_rect(self, frame_shape: Tuple[int, int, int]) -> Tuple[int, int, int, int]:
        height, width, _ = frame_shape
        region = self.channel_conf.get("region", {})
        x_pct = max(0, min(100, int(region.get("x", 0))))
        y_pct = max(0, min(100, int(region.get("y", 0))))
        w_pct = max(1, min(100, int(region.get("width", 100))))
        h_pct = max(1, min(100, int(region.get("height", 100))))

        x2_pct = min(100, x_pct + w_pct)
        y2_pct = min(100, y_pct + h_pct)

        x1 = int(width * x_pct / 100)
        y1 = int(height * y_pct / 100)
        x2 = max(x1 + 1, int(width * x2_pct / 100))
        y2 = max(y1 + 1, int(height * y2_pct / 100))
        return x1, y1, x2, y2

    def _extract_region(self, frame: cv2.Mat) -> Tuple[cv2.Mat, Tuple[int, int, int, int]]:
        x1, y1, x2, y2 = self._region_rect(frame.shape)
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    def _motion_detected(self, roi_frame: cv2.Mat) -> bool:
        if self.detection_mode != "motion":
            return True

        return self.motion_detector.update(roi_frame)

    @staticmethod
    def _offset_detections(detections: list[dict], roi_rect: Tuple[int, int, int, int]) -> list[dict]:
        x1, y1, _, _ = roi_rect
        adjusted: list[dict] = []
        for det in detections:
            box = det.get("bbox")
            if not box:
                continue
            det_copy = det.copy()
            det_copy["bbox"] = [int(box[0] + x1), int(box[1] + y1), int(box[2] + x1), int(box[3] + y1)]
            adjusted.append(det_copy)
        return adjusted

    async def _process_events(
        self, storage: AsyncEventDatabase, source: str, results: list[dict], channel_name: str
    ) -> None:
        for res in results:
            if res.get("unreadable"):
                logger.debug(
                    "Канал %s: номер помечен как нечитаемый (confidence=%.2f)",
                    channel_name,
                    res.get("confidence", 0.0),
                )
                continue
            if res.get("text"):
                event = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "channel": channel_name,
                    "plate": res.get("text", ""),
                    "confidence": res.get("confidence", 0.0),
                    "source": source,
                }
                event["id"] = await storage.insert_event_async(
                    channel=event["channel"],
                    plate=event["plate"],
                    confidence=event["confidence"],
                    source=event["source"],
                    timestamp=event["timestamp"],
                )
                self.event_ready.emit(event)
                logger.info(
                    "Канал %s: зафиксирован номер %s (conf=%.2f, track=%s)",
                    event["channel"],
                    event["plate"],
                    event["confidence"],
                    res.get("track_id", "-"),
                )

    async def _loop(self) -> None:
        pipeline, detector = await asyncio.to_thread(self._build_pipeline)
        storage = AsyncEventDatabase(self.db_path)

        source = str(self.channel_conf.get("source", "0"))
        capture = await asyncio.to_thread(self._open_capture, source)
        if capture is None:
            self.status_ready.emit(self.channel_conf.get("name", "Канал"), "Нет сигнала")
            logger.warning("Не удалось открыть источник %s для канала %s", source, self.channel_conf)
            return

        channel_name = self.channel_conf.get("name", "Канал")
        logger.info("Канал %s запущен (источник=%s)", channel_name, source)
        waiting_for_motion = False
        while self._running:
            ret, frame = await asyncio.to_thread(capture.read)
            if not ret:
                self.status_ready.emit(channel_name, "Поток остановлен")
                logger.warning("Поток остановлен для канала %s", channel_name)
                break

            roi_frame, roi_rect = self._extract_region(frame)
            motion_detected = self._motion_detected(roi_frame)

            if not motion_detected:
                if not waiting_for_motion and self.detection_mode == "motion":
                    self.status_ready.emit(channel_name, "Ожидание движения")
                waiting_for_motion = True
            else:
                if waiting_for_motion:
                    self.status_ready.emit(channel_name, "Движение обнаружено")
                waiting_for_motion = False
                if self._inference_limiter.allow():
                    detections = await asyncio.to_thread(detector.track, roi_frame)
                    detections = self._offset_detections(detections, roi_rect)
                    results = await asyncio.to_thread(pipeline.process_frame, frame, detections)
                    await self._process_events(storage, source, results, channel_name)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width, channel = rgb_frame.shape
            bytes_per_line = 3 * width
            # Копируем буфер, чтобы предотвратить обращение Qt к уже освобожденной памяти
            # во время перерисовок окна.
            q_image = QtGui.QImage(
                rgb_frame.data, width, height, bytes_per_line, QtGui.QImage.Format_RGB888
            ).copy()
            self.frame_ready.emit(channel_name, q_image)

        capture.release()

    def run(self) -> None:
        try:
            asyncio.run(self._loop())
        except Exception as exc:  # noqa: BLE001
            self.status_ready.emit(self.channel_conf.get("name", "Канал"), f"Ошибка: {exc}")
            logger.exception("Канал %s аварийно остановлен", self.channel_conf.get("name", "Канал"))

    def stop(self) -> None:
        self._running = False
