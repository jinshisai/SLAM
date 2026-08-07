# import model
from pathlib import Path

from ._pvanalysis import PVAnalysis
from . import analysis_tools
from . import fitfuncs


__version__ = (
    Path(__file__).resolve().parents[1] / 'VERSION'
).read_text().strip()

__all__ = ['PVAnalysis', 'analysis_tools', 'fitfuncs']
