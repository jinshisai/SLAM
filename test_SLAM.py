import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "slam-matplotlib")
)

import numpy as np

from channelfit import ChannelFit
from pvanalysis import PVAnalysis
from pvfitting.mockpvd import MockPVD
from pvfitting import PVFitting
from velgrad import VelGrad


DATA = Path(__file__).parent / "testfits"
CENTER = "04h39m53.878s +26d03m09.43s"


def test_import():
    a = [ChannelFit, PVAnalysis, PVFitting, VelGrad]
    assert None not in a


def test_pvanalysis_edgeridge_workflow(tmp_path):
    outname = tmp_path / "pvanalysis"
    impv = PVAnalysis(
        DATA / "test.pvanalysis.fits",
        rms=1.7e-3,
        vsys=6.4,
        dist=140.0,
        incl=48.0,
    )

    impv.get_edgeridge(
        str(outname),
        thr=5.0,
        ridgemode="mean",
        use_position=True,
        use_velocity=True,
        Mlim=[0, 10],
        xlim=np.array([-200, 0, 0, 200]) / 140.0,
        vlim=np.array([-5, 0, 0, 5]) + 6.4,
    )
    impv.write_edgeridge(str(outname))

    assert (tmp_path / "pvanalysis.edge.dat").exists()
    assert (tmp_path / "pvanalysis.ridge.dat").exists()
    assert len(impv.results_filtered["ridge"]["xcut"]["red"]) == 4
    assert len(impv.results_filtered["ridge"]["vcut"]["red"]) == 4


def test_velgrad_center_extraction():
    vg = VelGrad()
    vg.read_cubefits(
        cubefits=DATA / "test.cube.fits",
        center=CENTER,
        vsys=5.9,
        dist=140.0,
        sigma=1.7e-3,
        xmin=-200,
        xmax=200,
        ymin=-200,
        ymax=200,
        vmin=-3.6,
        vmax=3.6,
    )
    vg.get_2Dcenter(cutoff=5.0, vmask=[-2.0, 2.0], method="mean")

    assert vg.data.shape == (len(vg.v), len(vg.y), len(vg.x))
    assert vg.center["xc"].shape == vg.v.shape
    assert vg.center["yc"].shape == vg.v.shape
    assert np.count_nonzero(np.isfinite(vg.center["xc"])) > 0
    assert np.count_nonzero(np.isfinite(vg.center["yc"])) > 0


def test_pvfitting_loads_pv_and_generates_mock_diagrams():
    pvfit = PVFitting()
    pvfit.put_PV(
        pvmajorfits=DATA / "test.pvfitting.major.fits",
        pvminorfits=DATA / "test.pvfitting.minor.fits",
        dist=140.0,
        vsys=5.9,
        rmax=200.0,
        vmin=-3.6,
        vmax=3.6,
        sigma=1.7e-3,
        skipto=5,
    )

    mock = MockPVD(
        pvfit.x,
        pvfit.x,
        pvfit.v,
        nsubgrid=1,
        nnest=[2],
        beam=pvfit.beam,
        reslim=10,
        signmajor=-1,
        signminor=1,
        pa_major=2.0 - 180.0,
        pa_minor=92.0 - 180.0,
    )
    major, minor = mock.generate_mockpvd(
        Mstar=0.5,
        Rc=100.0,
        alphainfall=0.6,
        taumax=1.0,
        frho=10.0,
        incl=85.0,
        rout=300.0,
        axis="both",
    )

    assert pvfit.dpvmajor.shape == (len(pvfit.v), len(pvfit.x))
    assert pvfit.dpvminor.shape == pvfit.dpvmajor.shape
    assert major.shape == pvfit.dpvmajor.shape
    assert minor.shape == pvfit.dpvminor.shape
    assert np.nanmax(major) > 0
    assert np.nanmax(minor) > 0


def test_channelfit_makegrid_builds_deterministic_state():
    chan = ChannelFit(scaling="uniform", progressbar=False)
    chan.makegrid(
        cubefits=DATA / "test.cube.fits",
        center=CENTER,
        pa=2.0,
        incl=85.0,
        vsys=5.9,
        dist=140.0,
        sigma=1.7e-3,
        rmax=200.0,
        vlim=[-3.6, -2.0, 2.0, 3.6],
        nlayer=2,
        skipto=5,
    )

    assert chan.data_valid.shape[0] == len(chan.v_valid)
    assert chan.mom0.shape == (len(chan.y), len(chan.x))
    assert chan.mom1.shape == chan.mom0.shape
    assert chan.gaussbeam.ndim == 2
    assert chan.xnest.shape[0] == 2
    assert np.count_nonzero(np.isfinite(chan.mom0)) > 0
