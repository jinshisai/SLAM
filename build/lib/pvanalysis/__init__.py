# import model
from importlib.metadata import version
from pathlib import Path

from ._pvanalysis import PVAnalysis
from . import analysis_tools
from . import fitfuncs


_version_file = Path(__file__).resolve().parents[1] / 'VERSION'
__version__ = (
    _version_file.read_text().strip()
    if _version_file.is_file()
    else version('slam-astro')
)

__all__ = ['PVAnalysis', 'analysis_tools', 'fitfuncs']
