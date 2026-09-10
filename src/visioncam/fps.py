"""Medição de FPS (frames por segundo) do laço de captura."""

from __future__ import annotations

import time
from collections import deque


class FpsMeter:
    """Calcula o FPS por média móvel sobre os últimos N frames.

    Medir o intervalo entre dois frames isolados produz um número que
    oscila demais para ser lido na tela. Guardando os instantes dos
    últimos ``window`` frames e dividindo a quantidade de intervalos pelo
    tempo total, o valor fica estável sem deixar de reagir a quedas de
    desempenho.

    >>> medidor = FpsMeter(window=3)
    >>> medidor.value is None  # ainda sem amostras suficientes
    True
    """

    def __init__(self, window: int = 30, clock=time.perf_counter) -> None:
        if window < 2:
            raise ValueError("window precisa ser >= 2 para existir ao menos um intervalo")
        self._timestamps: deque[float] = deque(maxlen=window)
        self._clock = clock

    def tick(self) -> float | None:
        """Registra a chegada de um frame e devolve o FPS atual."""
        self._timestamps.append(self._clock())
        return self.value

    @property
    def value(self) -> float | None:
        """FPS médio, ou ``None`` enquanto não houver dois frames."""
        if len(self._timestamps) < 2:
            return None
        elapsed = self._timestamps[-1] - self._timestamps[0]
        if elapsed <= 0:
            return None
        return (len(self._timestamps) - 1) / elapsed

    def reset(self) -> None:
        self._timestamps.clear()
