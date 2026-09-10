"""Detecção de bordas com o algoritmo de Canny — visão computacional clássica.

Sem redes neurais: só convoluções e limiares. É o melhor lugar para entender
o pipeline básico de processamento de imagem, porque cada etapa tem um efeito
visível quando você mexe nos parâmetros.
"""

from __future__ import annotations

import cv2
import numpy as np

from .base import Processor, Result


class EdgesProcessor(Processor):
    key = "bordas"
    name = "Bordas (Canny)"
    description = "Escala de cinza -> desfoque -> gradiente -> limiares de Canny"

    def __init__(self, low: int = 80, high: int = 160, blur: int = 5, overlay: bool = True) -> None:
        # O desfoque precisa de kernel ímpar; arredondar aqui evita um erro
        # confuso vindo lá de dentro do OpenCV.
        self.blur = blur if blur % 2 == 1 else blur + 1
        self.low = low
        self.high = high
        self.overlay = overlay

    def process(self, frame: np.ndarray) -> Result:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # O desfoque gaussiano remove ruído que, sem ele, viraria "borda falsa".
        blurred = cv2.GaussianBlur(gray, (self.blur, self.blur), 0)
        edges = cv2.Canny(blurred, self.low, self.high)

        if self.overlay:
            # Pinta as bordas em ciano sobre o frame original.
            output = frame.copy()
            output[edges > 0] = (255, 220, 60)
        else:
            output = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

        ratio = float(np.count_nonzero(edges)) / edges.size
        return Result(
            frame=output,
            stats=[
                f"Limiares: {self.low}/{self.high}  Blur: {self.blur}",
                f"Pixels de borda: {ratio * 100:.1f}%",
            ],
        )
