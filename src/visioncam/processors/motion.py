"""Detecção de movimento por subtração de fundo (MOG2).

O MOG2 aprende, pixel a pixel, um modelo estatístico do que é "fundo" ao
longo dos últimos frames. O que não se encaixa nesse modelo vira frente
(foreground) — ou seja, algo que se moveu. Depois disso é processamento de
imagem clássico: limpar o ruído com morfologia e agrupar os pixels restantes
em contornos.
"""

from __future__ import annotations

import cv2
import numpy as np

from ..hud import GOOD, draw_box
from .base import Processor, Result


class MotionProcessor(Processor):
    key = "movimento"
    name = "Movimento"
    description = "Subtração de fundo MOG2 + contornos das regiões em movimento"

    def __init__(self, min_area: int = 900, history: int = 300, var_threshold: int = 40) -> None:
        self.min_area = min_area
        self._history = history
        self._var_threshold = var_threshold
        self._subtractor: cv2.BackgroundSubtractorMOG2 | None = None
        self._kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    def setup(self) -> None:
        self._subtractor = cv2.createBackgroundSubtractorMOG2(
            history=self._history,
            varThreshold=self._var_threshold,
            detectShadows=False,  # sombras viram ruído cinza; sem elas o resultado é mais limpo
        )

    def process(self, frame: np.ndarray) -> Result:
        if self._subtractor is None:
            self.setup()
        assert self._subtractor is not None

        mask = self._subtractor.apply(frame)
        # Abertura remove pontinhos soltos; dilatação junta pedaços do mesmo objeto.
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)
        mask = cv2.dilate(mask, self._kernel, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        output = frame.copy()

        detected = 0
        largest = 0.0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < self.min_area:
                continue
            detected += 1
            largest = max(largest, area)
            draw_box(output, cv2.boundingRect(contour), f"{int(area)} px", color=GOOD)

        moving_ratio = float(np.count_nonzero(mask)) / mask.size
        return Result(
            frame=output,
            stats=[
                f"Objetos em movimento: {detected}",
                f"Área em movimento: {moving_ratio * 100:.1f}%",
                f"Maior região: {int(largest)} px",
            ],
        )

    def close(self) -> None:
        self._subtractor = None
