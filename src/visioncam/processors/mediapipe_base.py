"""Base compartilhada pelos processadores que usam MediaPipe Tasks.

Três detalhes se repetiriam em todos eles e por isso moram aqui:

1. **Import tardio.** O MediaPipe só é importado dentro de ``setup()``. Assim
   é possível listar os modos, rodar os testes e usar os modos clássicos sem
   ter a biblioteca instalada.
2. **Conversão de cor.** O OpenCV entrega BGR; o MediaPipe espera RGB.
3. **Timestamps.** No modo VIDEO cada chamada precisa de um timestamp em
   milissegundos estritamente crescente, senão o MediaPipe recusa o frame.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .. import models
from .base import Processor, ProcessorUnavailable


class MediaPipeProcessor(Processor):
    """Cuida do ciclo de vida de um detector do MediaPipe Tasks."""

    #: Chave do modelo em :data:`visioncam.models.MODELS`.
    model_key: str = ""

    def __init__(self, models_dir: Path | None = None, auto_download: bool = False) -> None:
        self._models_dir = models_dir
        self._auto_download = auto_download
        self._detector: Any = None
        self._start: float | None = None
        self._last_timestamp_ms = -1

    # -- ciclo de vida -----------------------------------------------------

    def setup(self) -> None:
        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision
        except ImportError as exc:  # pragma: no cover - depende do ambiente
            raise ProcessorUnavailable(
                "MediaPipe não está instalado. Rode: pip install mediapipe"
            ) from exc

        try:
            model_path = models.ensure(self.model_key, self._models_dir, self._auto_download)
        except FileNotFoundError as exc:
            raise ProcessorUnavailable(str(exc)) from exc

        try:
            self._detector = self._create_detector(mp_python, vision, str(model_path))
        except Exception as exc:  # pragma: no cover - erro de runtime da lib
            raise ProcessorUnavailable(
                f"Falha ao carregar o modelo '{self.model_key}': {exc}"
            ) from exc

        self._start = time.perf_counter()
        self._last_timestamp_ms = -1

    def _create_detector(self, mp_python: Any, vision: Any, model_path: str) -> Any:
        raise NotImplementedError

    def close(self) -> None:
        if self._detector is not None:
            self._detector.close()
            self._detector = None

    # -- utilidades --------------------------------------------------------

    def _next_timestamp_ms(self) -> int:
        """Timestamp em ms garantidamente maior que o da chamada anterior."""
        elapsed_ms = int((time.perf_counter() - (self._start or 0.0)) * 1000)
        timestamp = max(elapsed_ms, self._last_timestamp_ms + 1)
        self._last_timestamp_ms = timestamp
        return timestamp

    def _to_mp_image(self, frame: np.ndarray) -> Any:
        import mediapipe as mp

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # O MediaPipe exige um buffer contíguo; cvtColor já devolve um, mas a
        # garantia explícita evita surpresas se alguém passar uma fatia.
        return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

    def _require_detector(self) -> Any:
        if self._detector is None:
            raise ProcessorUnavailable(
                f"Processador '{self.key}' usado sem setup(). Chame setup() antes de process()."
            )
        return self._detector


def to_pixels(landmarks: Any, width: int, height: int) -> list[tuple[int, int]]:
    """Converte landmarks normalizados (0..1) em coordenadas de pixel."""
    return [
        (int(landmark.x * width), int(landmark.y * height))
        for landmark in landmarks
    ]
