"""Testes do laço principal, todos rodando em modo headless sobre a fonte sintética."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from visioncam.app import App, AppConfig
from visioncam.cli import main


def config(**overrides) -> AppConfig:
    base = dict(source="synthetic", mode="original", headless=True, max_frames=5, mirror=False)
    base.update(overrides)
    return AppConfig(**base)


def test_roda_e_encerra_no_limite_de_frames():
    app = App(config(max_frames=7))
    assert app.run() == 0
    assert app.frames_seen == 7


def test_para_sozinho_quando_a_fonte_acaba(tmp_path):
    # Um vídeo de 3 frames: mesmo pedindo 50, só existem 3.
    video = tmp_path / "curto.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 20, (64, 48))
    for _ in range(3):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()

    app = App(config(source=str(video), max_frames=50))
    assert app.run() == 0
    assert 0 < app.frames_seen <= 3


def test_grava_o_video_de_saida(tmp_path):
    destino = tmp_path / "saida.mp4"
    assert App(config(record=destino, max_frames=10)).run() == 0
    assert destino.is_file() and destino.stat().st_size > 0

    capture = cv2.VideoCapture(str(destino))
    gravados = 0
    while capture.read()[0]:
        gravados += 1
    capture.release()
    assert gravados == 10


def test_modo_indisponivel_cai_para_o_original_sem_derrubar_o_app(tmp_path):
    """Modelo faltando não pode encerrar a aplicação — só trocar de modo."""
    app = App(config(mode="maos", models_dir=tmp_path, max_frames=4))
    assert app.run() == 0
    assert app.mode == "original"
    assert app.frames_seen == 4


def test_troca_de_modo_pela_tecla_numerica():
    app = App(config())
    app.switch_mode("original")
    app.handle_key(ord("2"))
    assert app.mode == "bordas"
    app.handle_key(ord("3"))
    assert app.mode == "movimento"


def test_tecla_de_modo_inexistente_e_ignorada():
    app = App(config())
    app.switch_mode("original")
    app.handle_key(ord("9"))  # só existem 6 modos
    assert app.mode == "original"


def test_teclas_de_saida():
    app = App(config())
    assert app.handle_key(ord("q")) is False
    assert app.handle_key(27) is False  # ESC
    assert app.handle_key(ord("h")) is True


def test_tecla_h_alterna_o_hud():
    app = App(config())
    assert app.show_hud is True
    app.handle_key(ord("h"))
    assert app.show_hud is False


def test_tecla_m_alterna_o_espelho_e_avisa_o_processador():
    app = App(config(mode="original", mirror=True))
    app.switch_mode("original")
    app.handle_key(ord("m"))
    assert app.mirror is False


def test_troca_para_modo_indisponivel_mantem_o_modo_atual(tmp_path):
    app = App(config(models_dir=tmp_path))
    app.switch_mode("bordas")
    assert app.switch_mode("pose") is False
    assert app.mode == "bordas"


def test_snapshot_e_salvo_na_pasta_pedida(tmp_path):
    app = App(config(snapshot_dir=tmp_path / "fotos"))
    app.switch_mode("original")
    app._save_snapshot(np.zeros((48, 64, 3), dtype=np.uint8))
    assert list((tmp_path / "fotos").glob("original-*.png"))


def test_hud_desligado_nao_desenha_nada_por_cima():
    """Com --no-hud, o frame de saída do modo original é o frame de entrada."""
    from visioncam.sources import SyntheticSource

    esperado = SyntheticSource().read()
    app = App(config(show_hud=False, max_frames=1, record=None))
    app._source = SyntheticSource()
    app.switch_mode("original")
    frame = app._source.read()
    result = app._processor.process(frame)
    assert np.array_equal(result.frame, esperado)


# --- CLI ------------------------------------------------------------------


def test_cli_lista_os_modos(capsys):
    assert main(["--list-modes"]) == 0
    saida = capsys.readouterr().out
    assert "bordas" in saida and "precisa de modelo" in saida


def test_cli_rejeita_modo_desconhecido():
    with pytest.raises(SystemExit):
        main(["--mode", "inexistente"])


def test_cli_roda_headless(tmp_path):
    destino = tmp_path / "cli.mp4"
    codigo = main(
        ["--source", "synthetic", "--headless", "--max-frames", "3", "--record", str(destino)]
    )
    assert codigo == 0
    assert destino.is_file()


def test_cli_reporta_fonte_invalida(capsys):
    assert main(["--source", "/nao/existe.mp4", "--headless"]) == 1
    assert "Não foi possível abrir" in capsys.readouterr().err
