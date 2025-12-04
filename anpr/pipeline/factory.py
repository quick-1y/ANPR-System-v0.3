from __future__ import annotations

from typing import Tuple

from anpr.config import ModelConfig
from anpr.detection.yolo_detector import YOLODetector
from anpr.pipeline.anpr_pipeline import ANPRPipeline
from anpr.recognition.crnn_recognizer import CRNNRecognizer


def build_components(best_shots: int, cooldown_seconds: int, min_confidence: float) -> Tuple[ANPRPipeline, YOLODetector]:
    """Создаёт независимые компоненты пайплайна (детектор, OCR и агрегация)."""

    detector = YOLODetector(ModelConfig.YOLO_MODEL_PATH, ModelConfig.DEVICE)
    recognizer = CRNNRecognizer(ModelConfig.OCR_MODEL_PATH, ModelConfig.DEVICE)
    pipeline = ANPRPipeline(
        recognizer,
        best_shots,
        cooldown_seconds,
        min_confidence=min_confidence,
    )
    return pipeline, detector
