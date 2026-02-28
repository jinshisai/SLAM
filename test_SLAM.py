from channelfit import ChannelFit
from pvanalysis import PVAnalysis
from pvfitting import PVFitting
from velgrad import VelGrad


def test_import():
    a = [ChannelFit, PVAnalysis, PVFitting, VelGrad]
    assert None not in a
