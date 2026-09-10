"""Testes da geometria pura: contagem de dedos e leitura de postura.

Estas funções são o coração "inteligente" do projeto e não dependem do
MediaPipe — recebem qualquer objeto com ``.x``/``.y``. Por isso dá para
testá-las com coordenadas escritas à mão, sem modelo e sem imagem.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from visioncam.processors.hands import extended_fingers, name_gesture
from visioncam.processors.pose import describe_posture


@dataclass
class Ponto:
    x: float
    y: float
    visibility: float = 1.0


def mao(dedos_estendidos: set[int]) -> list[Ponto]:
    """Monta 21 pontos de uma mão apontando para cima, no pulso em (0.5, 0.9).

    Cada dedo recebe pontos alinhados verticalmente; um dedo "dobrado" tem a
    ponta puxada de volta na direção do pulso.
    """
    pontos = [Ponto(0.5, 0.9)] * 21
    pontos = list(pontos)
    bases = {0: 0.30, 1: 0.42, 2: 0.50, 3: 0.58, 4: 0.66}  # x de cada dedo
    indices = {0: (1, 2, 3, 4), 1: (5, 6, 7, 8), 2: (9, 10, 11, 12),
               3: (13, 14, 15, 16), 4: (17, 18, 19, 20)}
    for dedo, (mcp, pip, dip, tip) in indices.items():
        x = bases[dedo]
        pontos[mcp] = Ponto(x, 0.80)
        pontos[pip] = Ponto(x, 0.70)
        pontos[dip] = Ponto(x, 0.62)
        # Estendido: ponta longe do pulso. Dobrado: ponta de volta para a palma.
        pontos[tip] = Ponto(x, 0.52) if dedo in dedos_estendidos else Ponto(x, 0.78)
    return pontos


def test_mao_aberta_conta_cinco_dedos():
    assert extended_fingers(mao({0, 1, 2, 3, 4})) == (True,) * 5


def test_punho_fechado_conta_zero_dedos():
    assert extended_fingers(mao(set())) == (False,) * 5


def test_apenas_o_indicador_estendido():
    assert extended_fingers(mao({1})) == (False, True, False, False, False)


def test_a_regra_independe_da_rotacao_da_mao():
    """Girar a mão 180° não pode mudar quantos dedos estão estendidos.

    É por isso que a regra compara *distâncias* e não coordenadas: uma regra
    do tipo "ponta acima da junta" quebraria com a mão de cabeça para baixo.
    """
    original = mao({1, 2})
    invertida = [Ponto(1 - ponto.x, 1 - ponto.y) for ponto in original]
    assert extended_fingers(invertida) == extended_fingers(original)


@pytest.mark.parametrize(
    ("dedos", "nome"),
    [
        ((False, False, False, False, False), "punho"),
        ((True, True, True, True, True), "mão aberta"),
        ((False, True, False, False, False), "apontando"),
        ((False, True, True, False, False), "paz"),
        ((True, False, False, False, False), "joinha"),
        ((False, True, False, False, True), "rock"),
    ],
)
def test_gestos_conhecidos_tem_nome(dedos, nome):
    assert name_gesture(dedos) == nome


def test_gesto_desconhecido_vira_contagem():
    assert name_gesture((True, True, True, False, False)) == "3 dedos"
    assert name_gesture((False, False, True, False, False)) == "1 dedo"


def corpo(punho_esq_y: float, punho_dir_y: float) -> list[Ponto]:
    """33 pontos com ombros em y=0.40 e quadris em y=0.70 (tronco = 0.30)."""
    pontos = [Ponto(0.5, 0.5) for _ in range(33)]
    pontos[11] = Ponto(0.40, 0.40)  # ombro esquerdo
    pontos[12] = Ponto(0.60, 0.40)  # ombro direito
    pontos[23] = Ponto(0.42, 0.70)  # quadril esquerdo
    pontos[24] = Ponto(0.58, 0.70)  # quadril direito
    pontos[15] = Ponto(0.30, punho_esq_y)
    pontos[16] = Ponto(0.70, punho_dir_y)
    return pontos


def test_dois_bracos_levantados():
    assert describe_posture(corpo(0.10, 0.10)) == "os dois braços levantados"


def test_um_braco_levantado():
    assert describe_posture(corpo(0.10, 0.85)) == "um braço levantado"


def test_bracos_abaixados():
    assert describe_posture(corpo(0.85, 0.85)) == "braços abaixados"


def test_bracos_na_horizontal_nao_contam_como_levantados():
    """Braços abertos na horizontal deixam os punhos quase na altura do ombro.

    Sem a margem de tolerância, uma diferença de milésimos já seria lida como
    "braço levantado" — foi exatamente o que aconteceu na primeira versão.
    """
    assert describe_posture(corpo(0.39, 0.38)) == "braços na altura dos ombros"


def test_tronco_degenerado_nao_quebra():
    pontos = corpo(0.5, 0.5)
    pontos[23] = Ponto(0.42, 0.40)
    pontos[24] = Ponto(0.58, 0.40)  # quadris na mesma altura dos ombros
    assert describe_posture(pontos) == "postura indefinida"
