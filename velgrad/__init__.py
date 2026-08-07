from pathlib import Path

from ._velgrad import VelGrad


__version__ = (
    Path(__file__).resolve().parents[1] / 'VERSION'
).read_text().strip()

__all__ = ['VelGrad']
