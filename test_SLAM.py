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
DIST = 140.0
SIGMA = 1.7e-3
RMAX = 200.0


def read_velgrad_test_cube(vg):
    vg.read_cubefits(
        cubefits=DATA / "test.cube.fits",
        center="04h39m53.878s +26d03m09.43s",
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
        cubefits=DATA / "test.cube.fits",
        center="04h39m53.878s +26d03m09.43s",
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


def test_pvfitting_check_modelgrid_without_nesting(capsys):
    pvfit = PVFitting()
    pvfit.x = np.linspace(-20.0, 20.0, 9)
    pvfit.v = np.linspace(-2.0, 2.0, 7)
    pvfit.beam = None

    pvfit.check_modelgrid(nsubgrid=1, n_nest=None)

    output = capsys.readouterr().out
    assert "Nesting level: 1" in output
    assert "l=0:" in output


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


def test_pvanalysis_posterior_outputs(tmp_path, monkeypatch, capsys):
    """Joint samples drive files, asymmetric reports and selected model curves."""
    import importlib
    from types import SimpleNamespace
    import matplotlib.pyplot as plt
    from pvanalysis.analysis_tools import doublepower_r, doublepower_v
    from pvanalysis._pvanalysis import kepler_mass

    module = importlib.import_module('pvanalysis._pvanalysis')
    obj = PVAnalysis.__new__(PVAnalysis)
    obj.xsign, obj.dist, obj.vsys = 1, 140., 6.4
    obj._PVAnalysis__unit = 140.
    obj._PVAnalysis__use_position = True
    obj._PVAnalysis__use_velocity = True
    points = np.array([[20., 80.], [3., 1.], [1., 1.], [.1, .1]])
    empty = np.empty((4, 0))
    obj.results_filtered = {
        method: {cut: {'red': points.copy(), 'blue': empty.copy()}
                 for cut in ('xcut', 'vcut')}
        for method in ('edge', 'ridge')}
    # Deliberately correlated samples and a maximum outside the central interval.
    samples = np.array([[30., 40., 50., 60., 70.],
                        [3., 2.5, 2., 1.5, 1.],
                        [.4, .45, .5, .55, .6],
                        [.2, .25, .3, .35, .4],
                        [-.1, -.05, 0., .05, .1]])
    maximum = np.array([90., 4., .7, .1, .2])
    low, mid, high = np.percentile(samples, [16, 50, 84], axis=1)
    lnp = np.arange(5.)

    def sampler(bounds, logprob, **kwargs):
        assert kwargs['simpleoutput'] is False
        assert kwargs['return_chain'] is True
        return [maximum, low, mid, high, samples, lnp]

    monkeypatch.setattr(module, 'emcee_corner', sampler)
    prefix = str(tmp_path / 'pvanalysis')
    result = obj.fit_edgeridge(outname=prefix, include_pin=True,
                              include_vsys=True, return_chain=True,
                              return_lnp=True, best='popt')
    paths = [Path(prefix + '.popt_' + suffix + '.txt') for suffix in ('e', 'r')]
    before = [path.read_bytes() for path in paths]
    for path in paths:
        np.testing.assert_allclose(np.loadtxt(path), np.array([maximum, low, mid, high]).T)
        assert '#Rows:Rb,Vb,p_in,dp,dVsys' in path.read_text()
    np.testing.assert_allclose(result['edge']['popt'], mid)
    np.testing.assert_allclose(result['edge']['perr'], (high-low)/2)
    np.testing.assert_array_equal(obj.chain['edge'], samples)
    np.testing.assert_array_equal(obj.lnp['ridge'], lnp)
    expected = []
    for params in samples.T:
        rin = doublepower_r(3., *params)
        vout = doublepower_v(80., *params)
        expected.append([kepler_mass(rin, 3.-params[4], 1.),
                         kepler_mass(80., vout-params[4], 1.),
                         kepler_mass(params[0], params[1], 1.)])
    np.testing.assert_allclose(obj.mass_samples['edge'], np.array(expected).T)
    np.testing.assert_allclose(obj.mass_summaries['edge']['pmid'],
                               np.median(expected, axis=0))
    capsys.readouterr()
    obj.output_fitresult(best='popt')
    output = capsys.readouterr().out
    assert 'best=popt' in output
    assert 'R_b   = 90.00 (-53.60, +0.00)' in output
    assert 'dp    = 0.100 (-0.000, +0.268)' in output
    assert 'v_sys = 6.600' in output
    assert f"M_b   = 1440.000" in output

    # Vectorized mass boundaries must match scalar evaluation for each cut mode.
    for position, velocity in [(True, True), (True, False), (False, True)]:
        obj._PVAnalysis__use_position = position
        obj._PVAnalysis__use_velocity = velocity
        scalar_masses = []
        for params in samples.T:
            rin = doublepower_r(3., *params) if position else 20.
            vin = 3. if position else doublepower_v(rin, *params)
            rout = 80. if velocity else doublepower_r(1., *params)
            vout = doublepower_v(rout, *params) if velocity else 1.
            scalar_masses.append([kepler_mass(rin, vin-params[4], 1.),
                                  kepler_mass(rout, vout-params[4], 1.),
                                  kepler_mass(params[0], params[1], 1.)])
        np.testing.assert_allclose(obj._masses('edge', samples), np.array(scalar_masses).T)
    obj._PVAnalysis__use_position = True
    obj._PVAnalysis__use_velocity = True

    # Exercise the complete plotting entry point without image I/O.
    class Plot:
        def __init__(self, **kwargs):
            self.fig, self.ax = plt.subplots()
        def add_color(self, **kwargs):
            pass
        def add_contour(self, **kwargs):
            pass
        def set_axis(self):
            pass
        def savefig(self, **kwargs):
            curves.append(self.ax.lines[-1].get_ydata().copy())
            plt.close(self.fig)

    curves = []
    monkeypatch.setattr(module, 'PVPlot', Plot)
    obj.rms = .1
    obj.fitsdata = SimpleNamespace(restfreq=1., beam=None, pa=0., data=None,
                                  vaxis=None, xaxis=None, multibeam=False, header={})
    for best in ('popt', 'pmid'):
        obj.plot_fitresult(best=best, show=False,
                           plotedgepoint=False, plotridgepoint=False)
        assert obj.avevsys == obj.fit_summaries['edge'][best][4]
        np.testing.assert_allclose(obj.rvlim['edge'][0][0],
                                   doublepower_r(3., *obj.fit_summaries['edge'][best]))
    assert not np.allclose(curves[0], curves[2], equal_nan=True)
    obj.output_fitresult(best='pmid')
    median_output = capsys.readouterr().out
    assert 'R_b   = 50.00 +/- 13.60 au' in median_output
    assert '(-' not in median_output
    assert 'M_b   = 200.000 +/-' in median_output
    # pmid reports masses at median parameters, not marginal mass medians.
    saved_mass_summaries = obj.mass_summaries
    obj.mass_summaries = {}
    obj.output_fitresult(best='pmid')
    assert capsys.readouterr().out == median_output
    obj.mass_summaries = saved_mass_summaries
    # Check the original local error propagation at both fitted boundaries.
    from pvanalysis.analysis_tools import (doublepower_r_error,
                                          doublepower_v_error, p_inout)
    from pvanalysis._pvanalysis import kepler_mass_error
    errors = (high - low) / 2
    params = [*mid, *errors]
    params_no_dvsys = [*params[:-1], 0.]
    rin = doublepower_r(3., *mid)
    vout = doublepower_v(80., *mid)
    m_in = kepler_mass(rin, 3.-mid[4], 1.)
    m_out = kepler_mass(80., vout-mid[4], 1.)
    dr = doublepower_r_error(3., *params_no_dvsys)
    power = p_inout(mid[2], mid[3], mid[1], abs(3.-mid[4]))
    dm_in = m_in * np.sqrt((dr/rin)**2 +
                           ((2.-1./power)*errors[4]/(3.-mid[4]))**2)
    dv = doublepower_v_error(80., *params_no_dvsys)
    dm_out = 2.*m_out*dv/abs(vout-mid[4])
    dm_b = kepler_mass_error(mid[0], mid[1], errors[0], errors[1], 1.)
    for label, mass, error in [('M_in', m_in, dm_in),
                               ('M_out', m_out, dm_out), ('M_b', 200., dm_b)]:
        assert f'{label:<5} = {mass:.3f} +/- {error:.3f} Msun' in median_output
    assert [path.read_bytes() for path in paths] == before

    # Fixed parameters occupy every summary and leave the opt-in chain unset.
    def fixed_sampler(bounds, logprob, **kwargs):
        chain = samples[:1]
        return [maximum[:1], low[:1], mid[:1], high[:1], chain]
    monkeypatch.setattr(module, 'emcee_corner', fixed_sampler)
    obj.fit_edgeridge(outname=prefix, include_dp=False)
    for name in ('popt', 'plow', 'pmid', 'phigh'):
        np.testing.assert_allclose(obj.fit_summaries['edge'][name][2:], [.5, 0., 0.])
    assert obj.chain is None
    obj.output_fitresult(best='popt')
    assert 'p_in  = 0.500 (-0.000, +0.000)' in capsys.readouterr().out

    import pytest
    with pytest.raises(ValueError, match='best'):
        obj.output_fitresult(best='unknown')
    for cut in ('xcut', 'vcut'):
        obj.results_filtered['ridge'][cut]['red'] = empty.copy()
    failed_prefix = str(tmp_path / 'failed')
    assert obj.fit_edgeridge(outname=failed_prefix) == -1
    assert not list(tmp_path.glob('failed.popt_*.txt'))
