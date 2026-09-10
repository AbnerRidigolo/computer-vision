"""Desenho do HUD (informações sobrepostas ao vídeo).

Todo desenho acontece aqui para que os processadores cuidem só de *analisar*
o frame, e não de estilizar a saída. As funções alteram o frame recebido no
lugar (in-place), como é convencional no OpenCV.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterable, Sequence

import cv2
import numpy as np

# Cores em BGR — a ordem que o OpenCV usa, ao contrário do RGB.
WHITE = (245, 245, 245)
ACCENT = (80, 200, 255)
GOOD = (120, 220, 140)
WARN = (80, 180, 250)
BAD = (90, 90, 235)

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def ascii_safe(text: str) -> str:
    """Remove acentos para o texto caber na fonte embutida do OpenCV.

    As fontes Hershey do ``cv2.putText`` só cobrem ASCII: um "ç" ou um "ã"
    sai como caractere quebrado e ainda desalinha a medição de largura feita
    por ``getTextSize``. Em vez de arrastar uma dependência de renderização de
    fontes (Pillow, FreeType) só por causa disso, o texto é transliterado num
    ponto único — aqui — e o resto do código continua escrito em português
    normal, com acentos.

    >>> ascii_safe("Resolução: 640x480")
    'Resolucao: 640x480'
    >>> ascii_safe("Mãos — 2 detectadas")
    'Maos - 2 detectadas'
    """
    replacements = {"—": "-", "–": "-", "“": '"', "”": '"', "’": "'"}
    for source, target in replacements.items():
        text = text.replace(source, target)
    decomposed = unicodedata.normalize("NFKD", text)
    without_marks = "".join(char for char in decomposed if not unicodedata.combining(char))
    return without_marks.encode("ascii", "replace").decode("ascii")


def draw_translucent_rect(
    frame: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int] = (20, 20, 20),
    alpha: float = 0.55,
) -> None:
    """Escurece um retângulo do frame para o texto ficar legível sobre ele.

    O truque é recortar a região, misturá-la com uma cor sólida via
    ``addWeighted`` e devolver o resultado ao mesmo lugar.
    """
    x1, y1 = top_left
    x2, y2 = bottom_right
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    overlay = np.full_like(roi, color, dtype=np.uint8)
    cv2.addWeighted(overlay, alpha, roi, 1 - alpha, 0, dst=roi)


def draw_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int] = WHITE,
    scale: float = 0.55,
    thickness: int = 1,
) -> None:
    """Escreve texto com um contorno escuro, legível sobre qualquer fundo.

    O contorno são quatro cópias pretas deslocadas em um pixel na diagonal, e
    não um traço mais grosso. O motivo é sutil: no OpenCV o avanço horizontal
    de cada glifo cresce junto com a espessura, então um contorno desenhado com
    ``thickness + 2`` fica progressivamente mais largo que o preenchimento e
    deixa um "fantasma" preto do último caractere para fora da palavra. Com
    todas as passadas na mesma espessura, os glifos coincidem exatamente.
    """
    text = ascii_safe(text)
    x, y = origin
    for offset_x, offset_y in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
        cv2.putText(
            frame, text, (x + offset_x, y + offset_y), _FONT, scale, (0, 0, 0),
            thickness, cv2.LINE_AA,
        )
    cv2.putText(frame, text, origin, _FONT, scale, color, thickness, cv2.LINE_AA)


def text_width(text: str, scale: float = 0.55, thickness: int = 1) -> int:
    """Largura em pixels que o texto ocupa, já contando o contorno escuro.

    O contorno de :func:`draw_text` são cópias deslocadas em 1 pixel, então a
    largura real é a do texto mais 1 pixel de cada lado.
    """
    return cv2.getTextSize(ascii_safe(text), _FONT, scale, thickness)[0][0] + 2


def draw_panel(
    frame: np.ndarray,
    lines: Sequence[str],
    origin: tuple[int, int] = (12, 12),
    scale: float = 0.55,
) -> tuple[int, int, int, int]:
    """Desenha um painel com uma linha de texto por item de ``lines``.

    Devolve o retângulo ocupado, ``(x, y, largura, altura)``, para quem quiser
    empilhar outro elemento logo abaixo — ou verificar que nada vazou dele.
    """
    if not lines:
        return (origin[0], origin[1], 0, 0)
    line_height = int(26 * (scale / 0.55))
    padding = 10
    widths = [text_width(line, scale) for line in lines]
    box_w = max(widths) + padding * 2
    box_h = line_height * len(lines) + padding * 2 - 6
    x, y = origin
    draw_translucent_rect(frame, (x, y), (x + box_w, y + box_h))
    for i, line in enumerate(lines):
        draw_text(frame, line, (x + padding, y + padding + line_height * i + 14), scale=scale)
    return (x, y, box_w, box_h)


def draw_box(
    frame: np.ndarray,
    box: tuple[int, int, int, int],
    label: str | None = None,
    color: tuple[int, int, int] = ACCENT,
    thickness: int = 2,
) -> None:
    """Desenha uma caixa ``(x, y, largura, altura)`` com rótulo opcional."""
    x, y, w, h = box
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    if label:
        text_w = text_width(label, 0.5)
        label_y = max(0, y - 22)
        draw_translucent_rect(
            frame, (x, label_y), (x + text_w + 12, label_y + 22), color=(15, 15, 15), alpha=0.65
        )
        draw_text(frame, label, (x + 6, label_y + 16), color=color, scale=0.5)


def draw_landmarks(
    frame: np.ndarray,
    points: Sequence[tuple[int, int]],
    connections: Iterable[tuple[int, int]] = (),
    point_color: tuple[int, int, int] = ACCENT,
    line_color: tuple[int, int, int] = (235, 235, 235),
    radius: int = 3,
) -> None:
    """Desenha um esqueleto: primeiro as ligações, depois os pontos por cima."""
    for start, end in connections:
        if start < len(points) and end < len(points):
            cv2.line(frame, points[start], points[end], line_color, 2, cv2.LINE_AA)
    for point in points:
        cv2.circle(frame, point, radius, point_color, -1, cv2.LINE_AA)


def draw_footer(frame: np.ndarray, text: str, scale: float = 0.5) -> None:
    """Faixa inferior com o lembrete de atalhos de teclado."""
    h, w = frame.shape[:2]
    bar_h = 30
    draw_translucent_rect(frame, (0, h - bar_h), (w, h), alpha=0.6)
    draw_text(frame, text, (12, h - 10), scale=scale, color=(210, 210, 210))
