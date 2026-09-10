"""Laço principal: captura -> processa -> desenha -> exibe.

Esta é a única parte do projeto que conhece janela, teclado e gravação. Os
processadores continuam sendo funções puras sobre frames, o que os mantém
fáceis de testar isoladamente.
"""

from __future__ import annotations

import datetime as dt
import os
import platform
import sys
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import processors
from .fps import FpsMeter
from .hud import BAD, draw_footer, draw_panel, draw_text, text_width
from .processors.base import Processor, ProcessorUnavailable
from .sources import FrameSource, open_source

WINDOW_TITLE = "VisionCam"

#: Quantos segundos uma mensagem de status fica visível na tela.
STATUS_DURATION = 4.0


@dataclass
class AppConfig:
    """Todas as opções de execução, vindas da linha de comando."""

    source: str = "0"
    mode: str = "original"
    width: int | None = None
    height: int | None = None
    mirror: bool = True
    show_hud: bool = True
    headless: bool = False
    max_frames: int | None = None
    record: Path | None = None
    snapshot_dir: Path = field(default_factory=lambda: Path("snapshots"))
    models_dir: Path | None = None
    download_models: bool = False
    loop_video: bool = False
    fps: float = 30.0


def display_available() -> bool:
    """Diz se existe um servidor gráfico utilizável.

    Chamar ``cv2.imshow`` sem display no Linux não levanta exceção: o processo
    é abortado pelo Qt. Como não dá para tratar isso com ``try/except``, a
    verificação precisa acontecer antes.
    """
    if platform.system() in {"Darwin", "Windows"}:
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


class App:
    """Orquestra fonte de vídeo, processador ativo, HUD e teclado."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.mode = config.mode
        self.mirror = config.mirror
        self.show_hud = config.show_hud
        self._processor: Processor | None = None
        self._source: FrameSource | None = None
        self._writer: cv2.VideoWriter | None = None
        self._fps = FpsMeter()
        self._status_lines: list[str] = []
        self._status_frames = 0
        self._status_error = False
        self._pending_snapshot = False
        self._frames_seen = 0
        self._snapshots = 0

    # -- processador ativo -------------------------------------------------

    def _processor_options(self) -> dict[str, object]:
        return {
            "models_dir": self.config.models_dir,
            "auto_download": self.config.download_models,
            "mirrored": self.mirror,
        }

    def switch_mode(self, key: str) -> bool:
        """Troca o modo ativo. Devolve ``False`` se o modo não estiver disponível.

        Quando a troca falha (modelo não baixado, por exemplo), o modo anterior
        continua rodando e o motivo aparece na tela — perder o vídeo inteiro
        por causa de um modo indisponível seria um péssimo negócio.
        """
        previous = self._processor
        try:
            candidate = processors.create(key, **self._processor_options())
            candidate.setup()
        except (ProcessorUnavailable, KeyError) as exc:
            self.set_status(str(exc), error=True)
            return False

        if previous is not None:
            previous.close()
        self._processor = candidate
        self.mode = key
        self.set_status(f"Modo: {candidate.name}")
        return True

    def set_status(self, message: str, error: bool = False) -> None:
        """Mostra uma mensagem temporária no rodapé.

        A duração é contada em frames, e não em segundos, para que o
        comportamento seja determinístico nos testes e no modo headless.
        """
        duration = STATUS_DURATION * (2 if error else 1)
        # Mensagens de erro do MediaPipe são longas; quebrar aqui, uma vez só,
        # evita refazer a conta a cada frame enquanto a mensagem estiver visível.
        self._status_lines = _wrap(message, 78)
        self._status_frames = int(duration * max(self.config.fps, 1))
        self._status_error = error

    # -- teclado -----------------------------------------------------------

    def handle_key(self, key: int) -> bool:
        """Trata uma tecla. Devolve ``False`` quando o app deve encerrar."""
        if key in (ord("q"), 27):  # 27 = ESC
            return False
        if key == ord("h"):
            self.show_hud = not self.show_hud
            self.set_status(f"HUD {'ligado' if self.show_hud else 'desligado'}")
        elif key == ord("m"):
            self.mirror = not self.mirror
            # O modo de mãos usa esta informação para rotular esquerda/direita.
            if hasattr(self._processor, "mirrored"):
                self._processor.mirrored = self.mirror
            self.set_status(f"Espelho {'ligado' if self.mirror else 'desligado'}")
        elif key == ord("s"):
            self._pending_snapshot = True
        elif ord("1") <= key <= ord("9"):
            index = key - ord("1")
            available = processors.keys()
            if index < len(available):
                self.switch_mode(available[index])
        return True

    # -- desenho -----------------------------------------------------------

    def _compose_hud(self, frame: np.ndarray, stats: list[str]) -> None:
        fps = self._fps.value
        lines = [
            f"FPS: {fps:.1f}" if fps else "FPS: --",
            f"Modo: {self._processor.name if self._processor else '--'}",
            *stats,
        ]
        draw_panel(frame, lines)

        # O rodapé precisa caber na largura do vídeo. Tenta a versão com o
        # nome de cada modo e, se não couber, cai para uma versão compacta.
        controls = "h HUD | m espelho | s foto | q sair"
        detailed = " | ".join(
            f"{i + 1} {processors.REGISTRY[key].name}" for i, key in enumerate(processors.keys())
        )
        compact = " | ".join(f"{i + 1} {key}" for i, key in enumerate(processors.keys()))
        for candidate in (f"{detailed} | {controls}", f"{compact} | {controls}",
                          f"1-{len(processors.keys())} modos | {controls}"):
            if text_width(candidate, 0.5) <= frame.shape[1] - 24:
                break
        draw_footer(frame, candidate)

    def _draw_status(self, frame: np.ndarray) -> None:
        if self._status_frames <= 0 or not self._status_lines:
            self._status_lines = []
            return
        self._status_frames -= 1
        color = BAD if self._status_error else (235, 235, 235)
        baseline = frame.shape[0] - 52
        for offset, chunk in enumerate(reversed(self._status_lines)):
            draw_text(frame, chunk, (12, baseline - offset * 20), color=color, scale=0.5)

    # -- gravação e snapshots ---------------------------------------------

    def _ensure_writer(self, frame: np.ndarray) -> None:
        if self.config.record is None or self._writer is not None:
            return
        self.config.record.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        height, width = frame.shape[:2]
        self._writer = cv2.VideoWriter(
            str(self.config.record), fourcc, self.config.fps, (width, height)
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Não foi possível abrir {self.config.record} para gravação.")

    def _save_snapshot(self, frame: np.ndarray) -> None:
        self.config.snapshot_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        path = self.config.snapshot_dir / f"{self.mode}-{stamp}.png"
        cv2.imwrite(str(path), frame)
        self._snapshots += 1
        self.set_status(f"Foto salva em {path}")

    # -- laço --------------------------------------------------------------

    def run(self) -> int:
        config = self.config
        if not config.headless and not display_available():
            print(
                "Nenhum display gráfico detectado. Rode com --headless "
                "(opcionalmente com --record saida.mp4) ou inicie uma sessão gráfica.",
                file=sys.stderr,
            )
            return 2

        self._source = open_source(config.source, config.width, config.height, config.loop_video)
        if not self.switch_mode(self.mode):
            # O modo pedido falhou: cai para o passthrough para o app continuar útil.
            if not self.switch_mode("original"):
                self._source.release()
                return 1

        window_open = False
        try:
            if not config.headless:
                cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
                window_open = True

            while config.max_frames is None or self._frames_seen < config.max_frames:
                frame = self._source.read()
                if frame is None:
                    break
                self._frames_seen += 1
                self._fps.tick()

                if self.mirror:
                    frame = cv2.flip(frame, 1)

                assert self._processor is not None
                try:
                    result = self._processor.process(frame)
                except ProcessorUnavailable as exc:
                    self.set_status(str(exc), error=True)
                    self.switch_mode("original")
                    continue

                output = result.frame
                if self.show_hud:
                    self._compose_hud(output, result.stats)
                self._draw_status(output)

                if self._pending_snapshot:
                    self._pending_snapshot = False
                    self._save_snapshot(output)

                self._ensure_writer(output)
                if self._writer is not None:
                    self._writer.write(output)

                if window_open:
                    cv2.imshow(WINDOW_TITLE, output)
                    if not self.handle_key(cv2.waitKey(1) & 0xFF):
                        break
                    # Fechar a janela no "X" também encerra o app.
                    if cv2.getWindowProperty(WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                        break
        finally:
            if self._processor is not None:
                self._processor.close()
            if self._writer is not None:
                self._writer.release()
            if self._source is not None:
                self._source.release()
            if window_open:
                cv2.destroyAllWindows()

        return 0

    # -- introspecção para testes -----------------------------------------

    @property
    def frames_seen(self) -> int:
        return self._frames_seen


def _wrap(text: str, width: int) -> list[str]:
    """Quebra o texto em linhas de no máximo ``width`` caracteres."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) <= width:
            current = candidate
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines or [""]
