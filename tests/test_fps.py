import pytest

from visioncam.fps import FpsMeter


class FakeClock:
    """Relógio controlado, para o teste não depender do tempo real."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


def test_janela_precisa_de_pelo_menos_dois_frames():
    with pytest.raises(ValueError):
        FpsMeter(window=1)


def test_sem_amostras_suficientes_o_valor_e_none():
    meter = FpsMeter(window=5, clock=FakeClock())
    assert meter.value is None
    assert meter.tick() is None


def test_calcula_fps_a_partir_dos_intervalos():
    clock = FakeClock()
    meter = FpsMeter(window=5, clock=clock)
    for _ in range(5):
        meter.tick()
        clock.now += 0.05  # 20 frames por segundo
    # Quatro intervalos de 50 ms entre os cinco timestamps guardados.
    assert meter.value == pytest.approx(20.0)


def test_janela_deslizante_descarta_amostras_antigas():
    clock = FakeClock()
    meter = FpsMeter(window=3, clock=clock)
    for _ in range(3):  # lento: 1 fps
        meter.tick()
        clock.now += 1.0
    for _ in range(3):  # rápido: 100 fps
        meter.tick()
        clock.now += 0.01
    assert meter.value == pytest.approx(100.0)


def test_reset_zera_as_amostras():
    clock = FakeClock()
    meter = FpsMeter(window=3, clock=clock)
    meter.tick()
    clock.now += 0.1
    meter.tick()
    assert meter.value is not None
    meter.reset()
    assert meter.value is None
