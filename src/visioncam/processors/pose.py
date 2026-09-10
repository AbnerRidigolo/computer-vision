"""Estimativa de pose: 33 pontos do corpo com o PoseLandmarker.

Cada ponto vem com um valor de *visibility* — a confiança de que aquela parte
do corpo realmente aparece no frame. Filtrar por essa confiança evita desenhar
membros que o modelo só está adivinhando (por exemplo, as pernas de alguém
sentado atrás de uma mesa).

Com os pontos em mãos, dá para derivar leituras simples de postura só
comparando alturas — é o que a função :func:`describe_posture` faz.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np

from ..hud import ACCENT
from .base import Result
from .landmarks import POSE_CONNECTIONS
from .mediapipe_base import MediaPipeProcessor, to_pixels

LEFT_SHOULDER, RIGHT_SHOULDER = 11, 12
LEFT_WRIST, RIGHT_WRIST = 15, 16
LEFT_HIP, RIGHT_HIP = 23, 24

#: Margem de tolerância, em frações do comprimento do tronco, para considerar
#: um punho "acima do ombro". Sem ela, uma diferença de poucos pixels — comum
#: com os braços abertos na horizontal — já seria lida como braço levantado.
SHOULDER_MARGIN = 0.25


def describe_posture(landmarks: Sequence[Any]) -> str:
    """Leitura simples da postura a partir da altura dos punhos.

    Lembre que, em coordenadas de imagem, o eixo Y cresce para baixo — então
    "punho acima do ombro" significa ``punho.y < ombro.y``.

    A comparação é feita em frações do comprimento do tronco (ombros até
    quadris) e não em pixels, para que o resultado não dependa da distância da
    pessoa até a câmera nem da resolução do vídeo.
    """
    shoulder_y = (landmarks[LEFT_SHOULDER].y + landmarks[RIGHT_SHOULDER].y) / 2
    hip_y = (landmarks[LEFT_HIP].y + landmarks[RIGHT_HIP].y) / 2
    torso = abs(hip_y - shoulder_y)
    if torso <= 0:  # pessoa de lado ou pontos degenerados: sem escala confiável
        return "postura indefinida"

    margin = SHOULDER_MARGIN * torso
    deltas = [
        (shoulder_y - landmarks[LEFT_WRIST].y) / torso,
        (shoulder_y - landmarks[RIGHT_WRIST].y) / torso,
    ]
    raised = sum(1 for delta in deltas if delta * torso > margin)
    if raised == 2:
        return "os dois braços levantados"
    if raised == 1:
        return "um braço levantado"
    if all(abs(delta * torso) <= margin for delta in deltas):
        return "braços na altura dos ombros"
    return "braços abaixados"


class PoseProcessor(MediaPipeProcessor):
    key = "pose"
    name = "Pose"
    description = "Esqueleto de 33 pontos e leitura de postura"
    model_key = "pose"

    def __init__(
        self,
        min_confidence: float = 0.5,
        min_visibility: float = 0.5,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.min_confidence = min_confidence
        self.min_visibility = min_visibility

    def _create_detector(self, mp_python: Any, vision: Any, model_path: str) -> Any:
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=self.min_confidence,
            min_tracking_confidence=self.min_confidence,
        )
        return vision.PoseLandmarker.create_from_options(options)

    def process(self, frame: np.ndarray) -> Result:
        detector = self._require_detector()
        result = detector.detect_for_video(self._to_mp_image(frame), self._next_timestamp_ms())

        output = frame.copy()
        height, width = frame.shape[:2]

        if not result.pose_landmarks:
            return Result(frame=output, stats=["Nenhuma pessoa detectada"])

        landmarks = result.pose_landmarks[0]
        points = to_pixels(landmarks, width, height)
        visible = [
            (landmark.visibility or 0.0) >= self.min_visibility for landmark in landmarks
        ]

        for start, end in POSE_CONNECTIONS:
            if visible[start] and visible[end]:
                cv2.line(output, points[start], points[end], (235, 235, 235), 2, cv2.LINE_AA)
        for point, is_visible in zip(points, visible):
            if is_visible:
                cv2.circle(output, point, 4, ACCENT, -1, cv2.LINE_AA)

        posture = describe_posture(landmarks)
        return Result(
            frame=output,
            stats=[
                f"Pontos visíveis: {sum(visible)}/{len(landmarks)}",
                f"Postura: {posture}",
            ],
        )
