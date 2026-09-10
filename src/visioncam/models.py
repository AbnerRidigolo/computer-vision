"""Download e cache dos modelos do MediaPipe Tasks.

Os arquivos de modelo pesam alguns megabytes e por isso não ficam no
repositório (veja ``.gitignore``). Este módulo sabe onde cada um mora e
baixa sob demanda, guardando tudo na pasta ``models/`` da raiz do projeto.

Uso pela linha de comando::

    python -m visioncam.models --list
    python -m visioncam.models --all
    python -m visioncam.models face hands
"""

from __future__ import annotations

import argparse
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path

_BASE = "https://storage.googleapis.com/mediapipe-models"


@dataclass(frozen=True)
class ModelSpec:
    """Um modelo baixável: chave curta, nome do arquivo e URL de origem."""

    key: str
    filename: str
    url: str
    description: str


MODELS: dict[str, ModelSpec] = {
    "face": ModelSpec(
        key="face",
        filename="blaze_face_short_range.tflite",
        url=f"{_BASE}/face_detector/blaze_face_short_range/float16/1/blaze_face_short_range.tflite",
        description="Detector de rostos BlazeFace (curta distância)",
    ),
    "hands": ModelSpec(
        key="hands",
        filename="hand_landmarker.task",
        url=f"{_BASE}/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        description="21 pontos por mão (HandLandmarker)",
    ),
    "pose": ModelSpec(
        key="pose",
        filename="pose_landmarker_lite.task",
        url=f"{_BASE}/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
        description="33 pontos do corpo (PoseLandmarker lite)",
    ),
}


def default_models_dir() -> Path:
    """Pasta ``models/`` na raiz do projeto (dois níveis acima deste arquivo)."""
    return Path(__file__).resolve().parents[2] / "models"


def model_path(key: str, models_dir: Path | None = None) -> Path:
    """Caminho local onde o modelo ``key`` deve estar."""
    spec = MODELS[key]
    return (models_dir or default_models_dir()) / spec.filename


def is_downloaded(key: str, models_dir: Path | None = None) -> bool:
    path = model_path(key, models_dir)
    return path.is_file() and path.stat().st_size > 0


def download(key: str, models_dir: Path | None = None, force: bool = False) -> Path:
    """Baixa o modelo se ainda não existir e devolve o caminho local."""
    spec = MODELS[key]
    target = model_path(key, models_dir)
    if target.is_file() and target.stat().st_size > 0 and not force:
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    # Escreve num arquivo temporário e só depois renomeia, para nunca deixar
    # um download interrompido no lugar de um modelo válido.
    temp = target.with_suffix(target.suffix + ".part")
    with urllib.request.urlopen(spec.url) as response, temp.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    temp.replace(target)
    return target


def ensure(key: str, models_dir: Path | None = None, auto_download: bool = False) -> Path:
    """Garante que o modelo esteja disponível localmente.

    Levanta ``FileNotFoundError`` com instruções claras quando o modelo não
    existe e o download automático não foi autorizado — falhar assim é bem
    melhor do que estourar um erro obscuro lá dentro do MediaPipe.
    """
    path = model_path(key, models_dir)
    if path.is_file() and path.stat().st_size > 0:
        return path
    if auto_download:
        return download(key, models_dir)
    raise FileNotFoundError(
        f"Modelo '{key}' não encontrado em {path}. "
        f"Baixe com: python -m visioncam.models {key}  (ou rode o app com --download-models)"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Baixa os modelos usados pelo VisionCam.")
    parser.add_argument("keys", nargs="*", help=f"modelos a baixar: {', '.join(MODELS)}")
    parser.add_argument("--all", action="store_true", help="baixa todos os modelos")
    parser.add_argument("--list", action="store_true", help="apenas lista os modelos e o status")
    parser.add_argument("--force", action="store_true", help="rebaixa mesmo se já existir")
    parser.add_argument("--dir", type=Path, default=None, help="pasta de destino")
    args = parser.parse_args(argv)

    unknown = [key for key in args.keys if key not in MODELS]
    if unknown:
        parser.error(
            f"modelo(s) desconhecido(s): {', '.join(unknown)}. Disponíveis: {', '.join(MODELS)}"
        )

    if args.list or (not args.all and not args.keys):
        for spec in MODELS.values():
            status = "OK " if is_downloaded(spec.key, args.dir) else "faltando"
            print(f"[{status:>8}] {spec.key:<6} {spec.description}")
        if not args.list:
            print("\nNada baixado. Use --all ou informe as chaves desejadas.")
        return 0

    keys = list(MODELS) if args.all else list(args.keys)
    for key in keys:
        print(f"Baixando {key} ({MODELS[key].description})...", flush=True)
        path = download(key, args.dir, force=args.force)
        print(f"  -> {path} ({path.stat().st_size / 1_048_576:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
