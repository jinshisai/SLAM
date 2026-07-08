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
CUBE = DATA / "test.cube.fits"
CENTER = "04h39m53.878s +26d03m09.43s"
DIST = 140.0
SIGMA = 1.7e-3
RMAX = 200.0


def read_velgrad_test_cube(vg):
    vg.read_cubefits(
        cubefits=CUBE,
        center=CENTER,
        vsys=5.9,
        dist=DIST,
        sigma=SIGMA,
        xmin=-RMAX,
        xmax=RMAX,
        ymin=-RMAX,
        ymax=RMAX,
        vmin=-3.6,
        vmax=3.6,
    )


def make_test_channelfit():
    chan = ChannelFit(scaling="uniform", progressbar=False)
    chan.makegrid(
        cubefits=CUBE,
        center=CENTER,
        pa=2.0,
        incl=85.0,
        vsys=5.9,
        dist=DIST,
        sigma=SIGMA,
        rmax=RMAX,
        vlim=[-3.6, -2.0, 2.0, 3.6],
        nlayer=2,
        skipto=5,
    )
    return chan


def test_import():
    assert all([ChannelFit, PVAnalysis, PVFitting, VelGrad])


def test_pvanalysis_edgeridge_workflow(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    monkeypatch.setattr(Figure, "savefig", lambda self, *args, **kwargs: None)

    outname = tmp_path / "pvanalysis"
    impv = PVAnalysis(
        DATA / "test.pvanalysis.fits",
        rms=SIGMA,
        vsys=6.4,
        dist=DIST,
        incl=48.0,
    )

    impv.get_edgeridge(
        str(outname),
        thr=5.0,
        ridgemode="mean",
        use_position=True,
        use_velocity=True,
        Mlim=[0, 10],
        xlim=np.array([-RMAX, 0, 0, RMAX]) / DIST,
        vlim=np.array([-5, 0, 0, 5]) + 6.4,
    )
    impv.write_edgeridge(str(outname))

    assert (tmp_path / "pvanalysis.edge.dat").exists()
    assert (tmp_path / "pvanalysis.ridge.dat").exists()
    ridge = impv.results_filtered["ridge"]
    assert all(len(ridge[cut]["red"]) == 4 for cut in ["xcut", "vcut"])


def test_velgrad_center_extraction():
    vg = VelGrad()
    read_velgrad_test_cube(vg)
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
        dist=DIST,
        vsys=5.9,
        rmax=RMAX,
        vmin=-3.6,
        vmax=3.6,
        sigma=SIGMA,
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
    chan = make_test_channelfit()

    assert chan.data_valid.shape[0] == len(chan.v_valid)
    assert chan.mom0.shape == (len(chan.y), len(chan.x))
    assert chan.mom1.shape == chan.mom0.shape
    assert chan.gaussbeam.ndim == 2
    assert chan.xnest.shape[0] == 2
    assert np.count_nonzero(np.isfinite(chan.mom0)) > 0
