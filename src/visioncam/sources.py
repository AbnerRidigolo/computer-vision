"""Fontes de frames: de onde o vídeo vem.

Todo o resto do programa só enxerga a interface :class:`FrameSource`, então
trocar a webcam por um arquivo de vídeo — ou por um gerador sintético usado
nos testes — não exige mudar uma linha do pipeline.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from typing import Protocol

import cv2
import numpy as np


class FrameSource(Protocol):
    """Contrato mínimo de uma fonte de vídeo."""

    def read(self) -> np.ndarray | None:
        """Devolve o próximo frame em BGR, ou ``None`` quando o vídeo acaba."""

    def release(self) -> None:
        """Libera os recursos (dispositivo, arquivo...)."""

    def describe(self) -> str:
        """Texto curto identificando a fonte, exibido no HUD."""


class CaptureSource:
    """Adaptador em volta de ``cv2.VideoCapture`` (webcam ou arquivo)."""

    def __init__(
        self,
        spec: int | str,
        width: int | None = None,
        height: int | None = None,
        loop: bool = False,
    ) -> None:
        self._spec = spec
        self._loop = loop
        self._capture = cv2.VideoCapture(spec)
        if not self._capture.isOpened():
            raise RuntimeError(
                f"Não foi possível abrir a fonte de vídeo {spec!r}. "
                "Verifique se a webcam está conectada, se outro programa não está "
                "usando o dispositivo, ou informe outro índice com --source."
            )
        if width:
            self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    def read(self) -> np.ndarray | None:
        ok, frame = self._capture.read()
        if not ok:
            if self._loop:
                # Arquivo chegou ao fim: volta para o primeiro frame.
                self._capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._capture.read()
            if not ok:
                return None
        return frame

    def release(self) -> None:
        self._capture.release()

    def describe(self) -> str:
        if isinstance(self._spec, int):
            return f"webcam #{self._spec}"
        return str(self._spec)


class SyntheticSource:
    """Gera frames animados sem precisar de câmera.

    Serve para dois propósitos: rodar os testes automatizados em qualquer
    máquina (inclusive em CI, onde não existe webcam) e permitir demonstrar
    o app em ambientes sem dispositivo de captura. A cena tem um fundo em
    gradiente estático e um círculo que orbita — assim o detector de
    movimento tem o que detectar.
    """

    def __init__(
        self, width: int = 640, height: int = 480, total_frames: int | None = None
    ) -> None:
        self.width = width
        self.height = height
        self._total = total_frames
        self._index = 0
        self._background = self._build_background(width, height)

    @staticmethod
    def _build_background(width: int, height: int) -> np.ndarray:
        gradient = np.linspace(30, 120, width, dtype=np.uint8)
        background = np.repeat(gradient[None, :], height, axis=0)
        return cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

    def read(self) -> np.ndarray | None:
        if self._total is not None and self._index >= self._total:
            return None
        frame = self._background.copy()
        angle = self._index * 0.12
        cx = int(self.width / 2 + math.cos(angle) * self.width * 0.28)
        cy = int(self.height / 2 + math.sin(angle) * self.height * 0.28)
        cv2.circle(frame, (cx, cy), 45, (60, 200, 255), thickness=-1)
        cv2.rectangle(frame, (40, 40), (140, 140), (200, 120, 60), thickness=-1)
        self._index += 1
        return frame

    def release(self) -> None:  # nada a liberar, mas cumpre o contrato
        return None

    def describe(self) -> str:
        return f"sintética {self.width}x{self.height}"


def open_source(
    spec: str,
    width: int | None = None,
    height: int | None = None,
    loop: bool = False,
) -> FrameSource:
    """Interpreta o valor de ``--source`` e devolve a fonte correspondente.

    - ``"0"``, ``"1"``, ...   índice de webcam
    - ``"synthetic"``         gerador sintético (não precisa de câmera)
    - qualquer outro texto    caminho de um arquivo de vídeo
    """
    if spec == "synthetic":
        return SyntheticSource(width or 640, height or 480)
    if spec.isdigit():
        return CaptureSource(int(spec), width, height, loop=loop)
    return CaptureSource(spec, width, height, loop=loop)


def iter_frames(source: FrameSource, limit: int | None = None) -> Iterator[np.ndarray]:
    """Itera sobre os frames de uma fonte, parando em ``limit`` se informado."""
    count = 0
    while limit is None or count < limit:
        frame = source.read()
        if frame is None:
            return
        count += 1
        yield frame
