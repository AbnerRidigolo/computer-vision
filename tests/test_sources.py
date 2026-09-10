import numpy as np
import pytest

from visioncam.sources import CaptureSource, SyntheticSource, iter_frames, open_source


def test_fonte_sintetica_gera_frames_bgr():
    source = SyntheticSource(width=320, height=240)
    frame = source.read()
    assert frame is not None
    assert frame.shape == (240, 320, 3)
    assert frame.dtype == np.uint8


def test_fonte_sintetica_anima_entre_frames():
    source = SyntheticSource()
    first, second = source.read(), source.read()
    assert not np.array_equal(first, second)


def test_fonte_sintetica_respeita_total_de_frames():
    source = SyntheticSource(total_frames=2)
    assert source.read() is not None
    assert source.read() is not None
    assert source.read() is None


def test_open_source_reconhece_sintetica_e_indice():
    assert isinstance(open_source("synthetic"), SyntheticSource)


def test_open_source_falha_com_mensagem_util_para_arquivo_inexistente():
    with pytest.raises(RuntimeError, match="Não foi possível abrir"):
        open_source("/caminho/que/nao/existe.mp4")


def test_iter_frames_respeita_o_limite():
    frames = list(iter_frames(SyntheticSource(), limit=4))
    assert len(frames) == 4


def test_iter_frames_para_quando_a_fonte_acaba():
    frames = list(iter_frames(SyntheticSource(total_frames=3), limit=10))
    assert len(frames) == 3


def test_capture_source_com_indice_invalido_levanta_erro():
    with pytest.raises(RuntimeError):
        CaptureSource(99)
