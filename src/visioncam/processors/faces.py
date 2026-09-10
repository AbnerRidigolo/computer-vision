"""Detecção de rostos com o BlazeFace (MediaPipe Tasks).

O modelo devolve, para cada rosto, uma caixa delimitadora, uma confiança e
seis pontos-chave: olhos, ponta do nariz, boca e as duas orelhas.
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from ..hud import ACCENT, GOOD, WARN, draw_box
from .base import Result
from .mediapipe_base import MediaPipeProcessor


def _confidence_color(score: float) -> tuple[int, int, int]:
    if score >= 0.8:
        return GOOD
    if score >= 0.6:
        return WARN
    return ACCENT


class FacesProcessor(MediaPipeProcessor):
    key = "rostos"
    name = "Rostos"
    description = "Detecção de rostos com caixas, confiança e pontos-chave"
    model_key = "face"

    def __init__(self, min_confidence: float = 0.5, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.min_confidence = min_confidence

    def _create_detector(self, mp_python: Any, vision: Any, model_path: str) -> Any:
        options = vision.FaceDetectorOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            min_detection_confidence=self.min_confidence,
        )
        return vision.FaceDetector.create_from_options(options)

    def process(self, frame: np.ndarray) -> Result:
        detector = self._require_detector()
        result = detector.detect_for_video(self._to_mp_image(frame), self._next_timestamp_ms())

        output = frame.copy()
        height, width = frame.shape[:2]
        scores: list[float] = []

        for detection in result.detections:
            box = detection.bounding_box
            score = detection.categories[0].score if detection.categories else 0.0
            scores.append(score)
            color = _confidence_color(score)
            draw_box(
                output,
                (box.origin_x, box.origin_y, box.width, box.height),
                f"rosto {score * 100:.0f}%",
                color=color,
            )
            # Os keypoints vêm normalizados (0..1), diferente da caixa.
            for keypoint in detection.keypoints or []:
                center = (int(keypoint.x * width), int(keypoint.y * height))
                cv2.circle(output, center, 2, color, -1, cv2.LINE_AA)

        stats = [f"Rostos detectados: {len(result.detections)}"]
        if scores:
            stats.append(f"Confiança média: {sum(scores) / len(scores) * 100:.0f}%")
        return Result(frame=output, stats=stats)
