"""Topologia dos esqueletos do MediaPipe (quais pontos ligam em quais).

Os índices são fixos e documentados pelo MediaPipe. Deixá-los aqui, em vez de
importar de dentro da biblioteca, mantém o desenho independente da versão
instalada e serve como referência rápida na hora de ler o código.
"""

from __future__ import annotations

# --- Mão: 21 pontos -------------------------------------------------------
# 0 = pulso; cada dedo tem 4 pontos, da base à ponta.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),          # polegar
    (0, 5), (5, 6), (6, 7), (7, 8),          # indicador
    (9, 10), (10, 11), (11, 12),             # médio
    (13, 14), (14, 15), (15, 16),            # anelar
    (0, 17), (17, 18), (18, 19), (19, 20),   # mindinho
    (5, 9), (9, 13), (13, 17),               # palma
)

#: Ponta e articulação intermediária de cada dedo, usadas para contar dedos.
FINGER_TIPS: tuple[int, ...] = (4, 8, 12, 16, 20)
FINGER_PIPS: tuple[int, ...] = (3, 6, 10, 14, 18)
WRIST = 0

FINGER_NAMES: tuple[str, ...] = ("polegar", "indicador", "médio", "anelar", "mindinho")

# --- Corpo: 33 pontos -----------------------------------------------------
POSE_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),  # rosto
    (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (15, 21), (17, 19),     # braço esquerdo
    (12, 14), (14, 16), (16, 18), (16, 20), (16, 22), (18, 20),               # braço direito
    (11, 23), (12, 24), (23, 24),                                             # tronco
    (23, 25), (25, 27), (27, 29), (29, 31), (27, 31),                         # perna esquerda
    (24, 26), (26, 28), (28, 30), (30, 32), (28, 32),                         # perna direita
)

#: Ponto de referência para o polegar: a base do indicador.
#: Medir o afastamento do polegar em relação à palma funciona melhor do que
#: medir em relação ao pulso, que fica perto demais do polegar dobrado.
THUMB_REFERENCE = 5
