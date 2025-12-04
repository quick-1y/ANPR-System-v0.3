from __future__ import annotations

from typing import Tuple

from detector import ANPR_Pipeline, CRNNRecognizer, YOLODetector, Config as ModelConfig


def build_components(best_shots: int, cooldown_seconds: int, min_confidence: float) -> Tuple[ANPR_Pipeline, YOLODetector]:
    """Создаёт независимые компоненты пайплайна (детектор, OCR и агрегация)."""

    detector = YOLODetector(ModelConfig.YOLO_MODEL_PATH, ModelConfig.DEVICE)
    recognizer = CRNNRecognizer(ModelConfig.OCR_MODEL_PATH, ModelConfig.DEVICE)
    pipeline = ANPR_Pipeline(
        recognizer,
        best_shots,
        cooldown_seconds,
        min_confidence=min_confidence,
    )
    return pipeline, detector
