"""Contrato comum a todos os processadores de frame."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import numpy as np


class ProcessorUnavailable(RuntimeError):
    """O processador não pôde ser inicializado (modelo ausente, lib faltando...).

    É um erro *esperado*: o app mostra a mensagem na tela e segue rodando,
    em vez de derrubar a aplicação inteira por causa de um modo indisponível.
    """


@dataclass
class Result:
    """Saída de um processador: o frame anotado e métricas para o HUD."""

    frame: np.ndarray
    stats: list[str] = field(default_factory=list)


class Processor(ABC):
    """Transforma um frame de entrada em um frame anotado.

    Cada modo do app (bordas, movimento, rostos...) é uma subclasse. O ciclo
    de vida é sempre o mesmo: ``setup()`` uma vez, ``process()`` a cada frame
    e ``close()`` no fim. ``setup()`` é separado do construtor de propósito —
    assim é possível listar todos os modos disponíveis sem carregar modelo
    nenhum na memória.
    """

    #: Identificador curto usado na CLI (``--mode``).
    key: str = "base"
    #: Nome exibido no HUD.
    name: str = "Base"
    #: Uma linha explicando o que o modo faz.
    description: str = ""

    def setup(self) -> None:  # noqa: B027 - gancho opcional: a maioria dos modos não usa
        """Carrega recursos pesados. Pode levantar :class:`ProcessorUnavailable`."""

    @abstractmethod
    def process(self, frame: np.ndarray) -> Result:
        """Recebe um frame BGR e devolve o resultado anotado."""

    def close(self) -> None:  # noqa: B027 - gancho opcional, idem
        """Libera recursos. Deve ser seguro chamar mais de uma vez."""

    def __enter__(self) -> Processor:
        self.setup()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
