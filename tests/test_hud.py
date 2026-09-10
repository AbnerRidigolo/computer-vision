import numpy as np
import pytest

from visioncam import hud


@pytest.mark.parametrize(
    ("entrada", "esperado"),
    [
        ("Resolução: 640x480", "Resolucao: 640x480"),
        ("Mãos — 2 detectadas", "Maos - 2 detectadas"),
        ("Confiança média", "Confianca media"),
        ("braços na altura dos ombros", "bracos na altura dos ombros"),
        ("já ASCII", "ja ASCII"),
    ],
)
def test_ascii_safe_remove_acentos(entrada, esperado):
    assert hud.ascii_safe(entrada) == esperado


def test_ascii_safe_e_idempotente():
    once = hud.ascii_safe("Área de detecção")
    assert hud.ascii_safe(once) == once


def test_text_width_cresce_com_o_texto():
    assert hud.text_width("ab") < hud.text_width("abcdefgh")


def test_text_width_ignora_acentos():
    # O texto é transliterado antes de ser medido, então as duas versões
    # precisam ter exatamente a mesma largura.
    assert hud.text_width("regiao") == hud.text_width("região")


def test_painel_cabe_dentro_da_propria_caixa(frame):
    """O texto não pode vazar para fora do retângulo escurecido.

    Compara a área do painel com o resto do frame: fora da caixa nada pode
    ter sido escrito.
    """
    canvas = np.full((200, 640, 3), 160, dtype=np.uint8)
    original = canvas.copy()
    lines = ["FPS: 60.0", "Postura: braços na altura dos ombros"]
    x, y, width, height = hud.draw_panel(canvas, lines, origin=(12, 12))
    assert width > 0 and height > 0

    # Restaura o miolo do painel: se sobrar qualquer diferença, é vazamento.
    fora_da_caixa = canvas.copy()
    fora_da_caixa[y : y + height, x : x + width] = original[y : y + height, x : x + width]
    assert np.array_equal(fora_da_caixa, original)


def test_painel_vazio_nao_altera_o_frame(frame):
    original = frame.copy()
    assert hud.draw_panel(frame, []) == (12, 12, 0, 0)
    assert np.array_equal(frame, original)


def test_retangulo_translucido_ignora_coordenadas_invertidas(frame):
    original = frame.copy()
    hud.draw_translucent_rect(frame, (80, 80), (10, 10))
    assert np.array_equal(frame, original)


def test_retangulo_translucido_recorta_nas_bordas(frame):
    # Não pode estourar: as coordenadas são cortadas para dentro do frame.
    hud.draw_translucent_rect(frame, (-50, -50), (10_000, 10_000))
    assert frame.shape == (120, 160, 3)


def test_draw_box_desenha_algo(frame):
    original = frame.copy()
    hud.draw_box(frame, (10, 40, 50, 30), "rosto 90%")
    assert not np.array_equal(frame, original)


def test_draw_landmarks_liga_os_pontos(frame):
    original = frame.copy()
    hud.draw_landmarks(frame, [(10, 10), (100, 100)], [(0, 1)])
    assert not np.array_equal(frame, original)


def test_draw_landmarks_ignora_conexoes_fora_do_alcance(frame):
    # Índice inexistente não pode derrubar o desenho.
    hud.draw_landmarks(frame, [(10, 10)], [(0, 5)])


@pytest.mark.parametrize(
    "texto",
    ["FPS: 60.0", "Postura: braços na altura dos ombros", "Objetos em movimento: 12"],
)
def test_draw_text_ocupa_exatamente_a_largura_medida(texto):
    """A tinta desenhada precisa caber na largura que :func:`text_width` promete.

    Este é o teste que pega a regressão do "fantasma": se o contorno for
    desenhado com espessura diferente da do preenchimento, o avanço de cada
    glifo muda, o contorno escorrega para a direita e a tinta passa a ocupar
    bem mais espaço do que foi medido.
    """
    canvas = np.full((60, 900, 3), 128, dtype=np.uint8)
    hud.draw_text(canvas, texto, (20, 40))

    pintado = np.any(canvas != 128, axis=(0, 2))
    colunas = np.flatnonzero(pintado)
    largura_real = int(colunas.max() - colunas.min() + 1)

    assert largura_real <= hud.text_width(texto)
