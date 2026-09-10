"""Modo mais simples possível: mostra o frame como veio da câmera.

Serve de referência de FPS — qualquer outro modo custa, no mínimo, o que
este custa. É também o menor exemplo possível de um :class:`Processor`.
"""

from __future__ import annotations

import numpy as np

from .base import Processor, Result


class PassthroughProcessor(Processor):
    key = "original"
    name = "Original"
    description = "Vídeo sem processamento — referência de FPS"

    def process(self, frame: np.ndarray) -> Result:
        return Result(frame=frame, stats=[f"Resolução: {frame.shape[1]}x{frame.shape[0]}"])
