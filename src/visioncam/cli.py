"""Interface de linha de comando do VisionCam."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, models, processors
from .app import App, AppConfig


def build_parser() -> argparse.ArgumentParser:
    modes = ", ".join(processors.keys())
    parser = argparse.ArgumentParser(
        prog="visioncam",
        description="Visão computacional em tempo real com OpenCV e MediaPipe.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Exemplos:\n"
            "  visioncam                                 # webcam padrão, modo original\n"
            "  visioncam --mode maos --download-models   # rastreia mãos, baixando o modelo\n"
            "  visioncam --source video.mp4 --loop       # roda em cima de um arquivo\n"
            "  visioncam --source synthetic --headless --max-frames 60 --record demo.mp4\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"visioncam {__version__}")
    parser.add_argument(
        "--source",
        default="0",
        help="índice da webcam (0, 1, ...), caminho de vídeo ou 'synthetic' (padrão: 0)",
    )
    parser.add_argument(
        "--mode", default="original", help=f"modo inicial: {modes} (padrão: original)"
    )
    parser.add_argument("--width", type=int, default=None, help="largura pedida à câmera")
    parser.add_argument("--height", type=int, default=None, help="altura pedida à câmera")
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="não espelha a imagem (por padrão o vídeo é espelhado, como um espelho de verdade)",
    )
    parser.add_argument("--no-hud", action="store_true", help="começa com o HUD desligado")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="processa sem abrir janela (útil em servidor ou CI, combine com --record)",
    )
    parser.add_argument(
        "--max-frames", type=int, default=None, help="encerra após N frames processados"
    )
    parser.add_argument(
        "--record", type=Path, default=None, help="grava a saída em um arquivo .mp4"
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        default=Path("snapshots"),
        help="pasta onde as fotos da tecla 's' são salvas",
    )
    parser.add_argument("--models-dir", type=Path, default=None, help="pasta dos modelos")
    parser.add_argument(
        "--download-models",
        action="store_true",
        help="baixa automaticamente o modelo que faltar, em vez de só avisar",
    )
    parser.add_argument("--loop", action="store_true", help="reinicia o vídeo ao chegar no fim")
    parser.add_argument("--fps", type=float, default=30.0, help="FPS usado na gravação")
    parser.add_argument(
        "--list-modes", action="store_true", help="lista os modos disponíveis e sai"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_modes:
        print("Modos disponíveis (a tecla é a posição na lista):\n")
        for index, (key, name, description) in enumerate(processors.describe(), start=1):
            needs = "" if key in processors.CLASSIC_KEYS else "  [precisa de modelo]"
            print(f"  {index}  {key:<10} {name:<16} {description}{needs}")
        print("\nModelos:")
        for spec in models.MODELS.values():
            status = "baixado" if models.is_downloaded(spec.key, args.models_dir) else "faltando"
            print(f"  {spec.key:<6} {status:<8} {spec.description}")
        return 0

    if args.mode not in processors.REGISTRY:
        parser.error(f"modo desconhecido: {args.mode}. Disponíveis: {', '.join(processors.keys())}")

    config = AppConfig(
        source=args.source,
        mode=args.mode,
        width=args.width,
        height=args.height,
        mirror=not args.no_mirror,
        show_hud=not args.no_hud,
        headless=args.headless,
        max_frames=args.max_frames,
        record=args.record,
        snapshot_dir=args.snapshot_dir,
        models_dir=args.models_dir,
        download_models=args.download_models,
        loop_video=args.loop,
        fps=args.fps,
    )

    try:
        return App(config).run()
    except RuntimeError as exc:
        print(f"Erro: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nEncerrado pelo usuário.")
        return 0


if __name__ == "__main__":
    sys.exit(main())
