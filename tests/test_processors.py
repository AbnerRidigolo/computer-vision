import numpy as np
import pytest

from visioncam import processors
from visioncam.processors import Processor, ProcessorUnavailable
from visioncam.processors.edges import EdgesProcessor
from visioncam.processors.motion import MotionProcessor
from visioncam.processors.passthrough import PassthroughProcessor
from visioncam.sources import SyntheticSource


def test_registro_lista_todos_os_modos():
    assert processors.keys() == ["original", "bordas", "movimento", "rostos", "maos", "pose"]


def test_todo_modo_tem_nome_e_descricao():
    for key, name, description in processors.describe():
        assert key and name and description


def test_todo_modo_e_subclasse_de_processor():
    assert all(issubclass(cls, Processor) for cls in processors.REGISTRY.values())


def test_create_com_chave_invalida_lista_as_validas():
    with pytest.raises(KeyError, match="original"):
        processors.create("inexistente")


def test_create_descarta_opcoes_que_o_modo_nao_aceita():
    """O app passa sempre o mesmo dicionário de opções para qualquer modo.

    Esta é a regressão que quebrou o modo de rostos: ``mirrored`` só existe no
    modo de mãos, e era repassado a todos.
    """
    processor = processors.create(
        "original", models_dir=None, auto_download=True, mirrored=True, min_confidence=0.9
    )
    assert isinstance(processor, PassthroughProcessor)


def test_create_repassa_opcoes_que_o_modo_aceita():
    processor = processors.create("bordas", low=10, high=20, mirrored=True)
    assert isinstance(processor, EdgesProcessor)
    assert (processor.low, processor.high) == (10, 20)


def test_accepted_options_enxerga_a_classe_base():
    # `models_dir` é declarado em MediaPipeProcessor, não em FacesProcessor.
    assert "models_dir" in processors.accepted_options(processors.FacesProcessor)
    assert "mirrored" in processors.accepted_options(processors.HandsProcessor)
    assert "mirrored" not in processors.accepted_options(processors.FacesProcessor)


def test_passthrough_devolve_o_frame_intacto(frame):
    result = PassthroughProcessor().process(frame)
    assert np.array_equal(result.frame, frame)
    assert any("160x120" in line for line in result.stats)


def test_bordas_mantem_o_formato_e_reporta_percentual(frame):
    result = EdgesProcessor().process(frame)
    assert result.frame.shape == frame.shape
    assert any("Pixels de borda" in line for line in result.stats)


def test_bordas_arredonda_o_kernel_para_impar():
    # Um kernel par faria o GaussianBlur estourar lá dentro do OpenCV.
    assert EdgesProcessor(blur=4).blur == 5


def test_bordas_sem_overlay_gera_imagem_binaria(frame):
    result = EdgesProcessor(overlay=False).process(frame)
    canais = [result.frame[:, :, i] for i in range(3)]
    assert all(np.array_equal(canais[0], canal) for canal in canais[1:])


def test_movimento_detecta_o_objeto_que_se_move():
    """A fonte sintética tem um círculo em órbita — ele precisa ser detectado."""
    source = SyntheticSource()
    processor = MotionProcessor(min_area=500)
    processor.setup()
    for _ in range(15):  # os primeiros frames servem para o MOG2 aprender o fundo
        result = processor.process(source.read())
    processor.close()

    detectados = next(line for line in result.stats if "Objetos em movimento" in line)
    assert int(detectados.split(":")[1]) >= 1


def test_movimento_ignora_ruido_menor_que_a_area_minima():
    source = SyntheticSource()
    processor = MotionProcessor(min_area=10**6)  # nada nesse frame é tão grande
    processor.setup()
    for _ in range(15):
        result = processor.process(source.read())
    processor.close()
    assert "Objetos em movimento: 0" in result.stats


def test_movimento_funciona_sem_setup_explicito(frame):
    # setup() é chamado sob demanda para o processador nunca quebrar por isso.
    assert MotionProcessor().process(frame).frame.shape == frame.shape


def test_context_manager_chama_setup_e_close(frame):
    with EdgesProcessor() as processor:
        assert processor.process(frame).frame.shape == frame.shape


def test_modo_com_modelo_ausente_avisa_em_vez_de_quebrar(tmp_path):
    processor = processors.create("rostos", models_dir=tmp_path, auto_download=False)
    with pytest.raises(ProcessorUnavailable, match="python -m visioncam.models"):
        processor.setup()


def test_usar_sem_setup_da_mensagem_clara(frame, tmp_path):
    processor = processors.create("pose", models_dir=tmp_path)
    with pytest.raises(ProcessorUnavailable, match="setup"):
        processor.process(frame)
