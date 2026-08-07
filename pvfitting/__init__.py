from pathlib import Path

from ._pvfitting import PVFitting
from .mockpvd import MockPVD


__version__ = (
    Path(__file__).resolve().parents[1] / 'VERSION'
).read_text().strip()

__all__ = ['PVFitting', 'MockPVD']
