"""VisionCam — visão computacional em tempo real com OpenCV e MediaPipe.

O pacote é dividido em peças pequenas e independentes para deixar claro
onde cada etapa do pipeline acontece:

``sources``     de onde vêm os frames (webcam, arquivo de vídeo ou sintético)
``processors``  o que é feito com cada frame (bordas, movimento, rostos, mãos, pose)
``hud``         como o resultado é desenhado na tela
``app``         o laço que amarra tudo e trata o teclado
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
