"""Registro dos modos disponíveis no app.

O registro é a única lista de modos que existe: a CLI, o HUD e os atalhos de
teclado são todos derivados dele. Para criar um modo novo basta escrever a
classe e adicioná-la em :data:`REGISTRY` — nenhum outro arquivo precisa mudar.
"""

from __future__ import annotations

import inspect
from typing import Any

from .base import Processor, ProcessorUnavailable, Result
from .edges import EdgesProcessor
from .faces import FacesProcessor
from .hands import HandsProcessor
from .motion import MotionProcessor
from .passthrough import PassthroughProcessor
from .pose import PoseProcessor

#: Modos na ordem em que aparecem no HUD (a posição define a tecla 1..N).
REGISTRY: dict[str, type[Processor]] = {
    processor.key: processor
    for processor in (
        PassthroughProcessor,
        EdgesProcessor,
        MotionProcessor,
        FacesProcessor,
        HandsProcessor,
        PoseProcessor,
    )
}

#: Modos que não dependem de MediaPipe nem de modelos baixados.
CLASSIC_KEYS: tuple[str, ...] = ("original", "bordas", "movimento")


def keys() -> list[str]:
    """Chaves dos modos, na ordem de exibição."""
    return list(REGISTRY)


def accepted_options(processor_class: type[Processor]) -> set[str]:
    """Nomes de parâmetros que o construtor do processador realmente aceita.

    Percorre toda a hierarquia porque as subclasses repassam ``**kwargs`` para
    a classe base — só olhar a assinatura mais específica não revelaria, por
    exemplo, o ``models_dir`` que vive em ``MediaPipeProcessor``.
    """
    names: set[str] = set()
    for klass in processor_class.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        for name, parameter in inspect.signature(init).parameters.items():
            if name == "self" or parameter.kind in (
                parameter.VAR_POSITIONAL,
                parameter.VAR_KEYWORD,
            ):
                continue
            names.add(name)
    return names


def create(key: str, **options: Any) -> Processor:
    """Instancia um modo pela chave, repassando só as opções que ele aceita.

    O app mantém um único dicionário de opções e o entrega a qualquer modo;
    filtrar aqui evita que cada chamador precise saber que ``mirrored`` só faz
    sentido para as mãos, ou que ``models_dir`` só vale para os modos com modelo.
    """
    try:
        processor_class = REGISTRY[key]
    except KeyError:
        raise KeyError(f"Modo desconhecido: {key!r}. Disponíveis: {', '.join(REGISTRY)}") from None

    accepted = accepted_options(processor_class)
    return processor_class(**{n: v for n, v in options.items() if n in accepted})


def describe() -> list[tuple[str, str, str]]:
    """Lista ``(chave, nome, descrição)`` de todos os modos."""
    return [(cls.key, cls.name, cls.description) for cls in REGISTRY.values()]


__all__ = [
    "CLASSIC_KEYS",
    "REGISTRY",
    "EdgesProcessor",
    "FacesProcessor",
    "HandsProcessor",
    "MotionProcessor",
    "PassthroughProcessor",
    "PoseProcessor",
    "Processor",
    "ProcessorUnavailable",
    "Result",
    "accepted_options",
    "create",
    "describe",
    "keys",
]
