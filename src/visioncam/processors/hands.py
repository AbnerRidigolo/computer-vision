"""Rastreamento de mãos, contagem de dedos e reconhecimento de gestos simples.

O MediaPipe devolve 21 pontos por mão. A partir deles dá para inferir bastante
coisa com geometria pura — sem treinar nada. A regra usada aqui é:

    um dedo está estendido quando a **ponta** está mais longe de um ponto de
    referência do que a **articulação intermediária**.

Para os quatro dedos longos a referência é o pulso; para o polegar, a base do
indicador (o polegar dobrado fica perto demais do pulso para a comparação
funcionar). Comparar distâncias, e não coordenadas, deixa a regra independente
da rotação da mão — funciona com a mão apontando para cima, para o lado ou de
cabeça para baixo.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import numpy as np

from ..hud import ACCENT, GOOD, draw_landmarks, draw_text
from .base import Result
from .landmarks import (
    FINGER_NAMES,
    FINGER_PIPS,
    FINGER_TIPS,
    HAND_CONNECTIONS,
    THUMB_REFERENCE,
    WRIST,
)
from .mediapipe_base import MediaPipeProcessor, to_pixels

#: Combinações de dedos estendidos com nome conhecido.
#: A ordem é (polegar, indicador, médio, anelar, mindinho).
GESTURES: dict[tuple[bool, ...], str] = {
    (False, False, False, False, False): "punho",
    (True, True, True, True, True): "mão aberta",
    (False, True, False, False, False): "apontando",
    (False, True, True, False, False): "paz",
    (True, False, False, False, False): "joinha",
    (True, False, False, False, True): "hang loose",
    (False, True, False, False, True): "rock",
    (True, True, False, False, True): "eu te amo",
    (False, False, False, False, True): "mindinho",
}


def _distance(a: Any, b: Any) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def extended_fingers(landmarks: Sequence[Any]) -> tuple[bool, ...]:
    """Diz, para cada um dos cinco dedos, se ele está estendido."""
    wrist = landmarks[WRIST]
    thumb_reference = landmarks[THUMB_REFERENCE]
    states = []
    for index, (tip, pip) in enumerate(zip(FINGER_TIPS, FINGER_PIPS, strict=True)):
        reference = thumb_reference if index == 0 else wrist
        states.append(_distance(landmarks[tip], reference) > _distance(landmarks[pip], reference))
    return tuple(states)


def name_gesture(fingers: tuple[bool, ...]) -> str:
    """Nome do gesto, ou a contagem de dedos quando não há um nome conhecido."""
    known = GESTURES.get(fingers)
    if known:
        return known
    total = sum(fingers)
    return f"{total} dedo" if total == 1 else f"{total} dedos"


class HandsProcessor(MediaPipeProcessor):
    key = "maos"
    name = "Mãos"
    description = "21 pontos por mão, contagem de dedos e gestos"
    model_key = "hands"

    def __init__(
        self,
        max_hands: int = 2,
        min_confidence: float = 0.5,
        mirrored: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.max_hands = max_hands
        self.min_confidence = min_confidence
        # Quando o frame é espelhado (modo selfie), a mão direita aparece como
        # esquerda para o modelo. A geometria continua correta; só o rótulo
        # precisa ser invertido na hora de mostrar ao usuário.
        self.mirrored = mirrored

    def _create_detector(self, mp_python: Any, vision: Any, model_path: str) -> Any:
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=self.max_hands,
            min_hand_detection_confidence=self.min_confidence,
            min_tracking_confidence=self.min_confidence,
        )
        return vision.HandLandmarker.create_from_options(options)

    def _display_label(self, raw_label: str) -> str:
        translation = {"Left": "esquerda", "Right": "direita"}
        if self.mirrored:
            raw_label = {"Left": "Right", "Right": "Left"}.get(raw_label, raw_label)
        return translation.get(raw_label, raw_label.lower())

    def process(self, frame: np.ndarray) -> Result:
        detector = self._require_detector()
        result = detector.detect_for_video(self._to_mp_image(frame), self._next_timestamp_ms())

        output = frame.copy()
        height, width = frame.shape[:2]
        stats = [f"Mãos detectadas: {len(result.hand_landmarks)}"]
        total_fingers = 0

        for landmarks, handedness in zip(result.hand_landmarks, result.handedness, strict=True):
            points = to_pixels(landmarks, width, height)
            fingers = extended_fingers(landmarks)
            total_fingers += sum(fingers)

            draw_landmarks(output, points, HAND_CONNECTIONS, point_color=ACCENT)
            # Destaca em verde só as pontas dos dedos estendidos.
            for is_up, tip in zip(fingers, FINGER_TIPS, strict=True):
                if is_up:
                    draw_landmarks(output, [points[tip]], point_color=GOOD, radius=6)

            label = self._display_label(handedness[0].category_name if handedness else "")
            wrist_x, wrist_y = points[WRIST]
            draw_text(
                output,
                f"{label}: {name_gesture(fingers)}",
                (max(4, wrist_x - 60), min(height - 8, wrist_y + 28)),
                color=GOOD,
                scale=0.6,
            )
            up_names = [n for n, is_up in zip(FINGER_NAMES, fingers, strict=True) if is_up]
            stats.append(f"  {label}: {', '.join(up_names) if up_names else 'nenhum dedo'}")

        if result.hand_landmarks:
            stats.insert(1, f"Total de dedos: {total_fingers}")
        return Result(frame=output, stats=stats)
