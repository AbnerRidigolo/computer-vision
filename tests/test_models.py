import pytest

from visioncam import models


def test_todos_os_modelos_tem_url_e_descricao():
    for key, spec in models.MODELS.items():
        assert spec.key == key
        assert spec.url.startswith("https://")
        assert spec.filename and spec.description


def test_model_path_usa_a_pasta_informada(tmp_path):
    caminho = models.model_path("face", tmp_path)
    assert caminho.parent == tmp_path
    assert caminho.name == models.MODELS["face"].filename


def test_is_downloaded_e_falso_em_pasta_vazia(tmp_path):
    assert not models.is_downloaded("face", tmp_path)


def test_is_downloaded_ignora_arquivo_vazio(tmp_path):
    # Um download interrompido não pode passar por modelo válido.
    models.model_path("face", tmp_path).touch()
    assert not models.is_downloaded("face", tmp_path)


def test_ensure_sem_download_explica_como_resolver(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"python -m visioncam\.models face"):
        models.ensure("face", tmp_path, auto_download=False)


def test_ensure_aceita_modelo_ja_presente(tmp_path):
    caminho = models.model_path("pose", tmp_path)
    caminho.write_bytes(b"conteudo qualquer")
    assert models.ensure("pose", tmp_path) == caminho


def test_cli_lista_status_sem_baixar_nada(tmp_path, capsys):
    assert models.main(["--list", "--dir", str(tmp_path)]) == 0
    saida = capsys.readouterr().out
    assert "faltando" in saida
    assert not list(tmp_path.iterdir())


def test_cli_rejeita_modelo_desconhecido(tmp_path):
    with pytest.raises(SystemExit):
        models.main(["fantasma", "--dir", str(tmp_path)])
