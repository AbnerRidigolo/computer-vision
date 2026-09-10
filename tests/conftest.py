"""Configuração comum dos testes."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def frame() -> np.ndarray:
    """Um frame BGR simples, com algum conteúdo para gerar bordas."""
    image = np.full((120, 160, 3), 40, dtype=np.uint8)
    image[30:90, 40:120] = (200, 180, 60)
    return image
