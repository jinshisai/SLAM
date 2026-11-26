import copy
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "slam-matplotlib")
)

import numpy as np
import pytest

from channelfit import ChannelFit
from pvanalysis import PVAnalysis
from pvfitting import PVFitting
from pvfitting.mockpvd import MockPVD
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


def make_test_pvanalysis():
    return PVAnalysis(DATA / "test.pvanalysis.fits", SIGMA, 6.4, DIST, incl=48.0)


def load_test_pvanalysis_points(tmp_path, methods=("edge", "ridge")):
    x = np.array([-150.0, -100.0, -50.0, 50.0, 100.0, 150.0])
    v = np.sign(x) * 2.0 * (np.abs(x) / 100.0) ** -1
    rows = np.concatenate(
        [
            np.c_[x, np.ones(6), v, np.zeros(6)],
            np.c_[x, np.zeros(6), v, np.full(6, 0.1)],
        ]
    )
    for method in ("edge", "ridge"):
        points = rows if method in methods else np.empty((0, 4))
        np.savetxt(tmp_path / f"points.{method}.dat", points, header="x dx v dv")
    obj = make_test_pvanalysis()
    obj.read_edgeridge(tmp_path / "points", quadrant="13")
    return obj


@pytest.fixture
def pv_sampler(monkeypatch):
    """Check fitting orchestration without paying for long production chains."""
    import pvanalysis._pvanalysis as module

    calls = []

    def sample(bounds, logprob, **kwargs):
        midpoint = np.mean(bounds, axis=0)
        assert np.isfinite(logprob(midpoint, *kwargs["args"]))
        calls.append(kwargs)
        output = [
            midpoint,
            midpoint,
            midpoint,
            midpoint,
            np.tile(midpoint[:, None], (1, 12)),
        ]
        if kwargs["return_lnp"]:
            output.append(np.zeros(12))
        return output

    monkeypatch.setattr(module, "emcee_corner", sample)
    return calls


def test_pvanalysis_edgeridge_workflow(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    monkeypatch.setattr(Figure, "savefig", lambda self, *args, **kwargs: None)

    outname = tmp_path / "pvanalysis"
    impv = make_test_pvanalysis()

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

    restored = make_test_pvanalysis()
    restored.read_edgeridge(outname)
    assert restored.xsign == impv.xsign
    for method in ("edge", "ridge"):
        for actual, expected in zip(restored._fit_data(method), impv._fit_data(method)):
            np.testing.assert_equal(actual, expected)


@pytest.mark.parametrize("methods", [(), ("edge",), ("ridge",), ("edge", "ridge")])
def test_pvanalysis_independent_and_empty_fits(tmp_path, methods, pv_sampler):
    obj = load_test_pvanalysis_points(tmp_path, methods)
    result = obj.fit_edgeridge(
        outname=tmp_path / "fit", save_corner=False, return_chain=True, return_lnp=True
    )
    assert len(pv_sampler) == len(methods)
    for method in ("edge", "ridge"):
        assert (result[method] is not None) == (method in methods)
        assert (tmp_path / f"fit.popt_{method[0]}.txt").exists() == (method in methods)
        if method in methods:
            assert np.loadtxt(tmp_path / f"fit.popt_{method[0]}.txt").shape == (8, 4)
            assert np.isfinite(obj.mass_samples[method]).all()
            assert obj.chain[method].shape[1] == 12
            assert obj.lnp[method].shape == (12,)
        else:
            assert obj.fit_status[method] == "no_points"
            assert obj.get_range()[method] is None
    obj.output_fitresult(best="pmid")
    obj.output_fitresult(best="popt")


def test_pvanalysis_separate_fits_options_and_invalidation(tmp_path, pv_sampler):
    obj = load_test_pvanalysis_points(tmp_path)
    options = {
        "range_corner": {"Rb": [20.0, 200.0]},
        "rangelevel": None,
        "labels": ["Radius", "Slope"],
        "nburnin": 10,
        "nsteps": 20,
        "figname": str(tmp_path / "disabled.png"),
    }
    obj.fit_edgeridge(
        methods=("edge",),
        include_dp=False,
        include_pin=True,
        outname=tmp_path / "edge",
        save_corner=False,
        return_chain=True,
        kwargs_emcee_corner=options,
    )
    forwarded = pv_sampler[0]
    assert forwarded["range_corner"] == [[20.0, 200.0], 0.8]
    assert forwarded["labels"] == ["Radius", "Slope"]
    assert forwarded["rangelevel"] is None and forwarded["nburnin"] == 10
    assert forwarded["nsteps"] == 20 and forwarded["figname"] is None
    assert options["range_corner"] == {"Rb": [20.0, 200.0]}
    assert options["figname"] == str(tmp_path / "disabled.png")
    edge, chain, mass = obj.popt["edge"], obj.chain["edge"], obj.mass_samples["edge"]
    obj.fit_edgeridge(
        methods=("ridge",),
        include_dp=False,
        outname=tmp_path / "ridge",
        save_corner=False,
        kwargs_emcee_corner={"nsteps": 30},
    )
    assert pv_sampler[1]["nsteps"] == 30
    assert obj.popt["edge"] is edge
    assert obj.chain["edge"] is chain
    assert obj.mass_samples["edge"] is mass
    obj.read_edgeridge(tmp_path / "points", methods=("ridge",))
    assert obj.popt == {"edge": None, "ridge": None}
    assert obj.chain == {"edge": None, "ridge": None}
    assert not obj.mass_samples and not obj.fit_summaries
    assert obj.point_status["edge"] == "disabled"


@pytest.mark.parametrize("cut", ["xcut", "vcut"])
@pytest.mark.parametrize("floor", [0.0, 0.5])
def test_pvanalysis_invalid_errors_and_empty_exports(tmp_path, cut, floor, pv_sampler):
    obj = make_test_pvanalysis()
    obj.get_edgeridge(
        str(tmp_path / "setup"),
        quadrant="13",
        Mlim=[],
        use_position=False,
        use_velocity=False,
    )
    x = np.linspace(-0.9, 0.9, 10)
    raw = np.array([x, obj.vsys + 3.0 * x, np.zeros(10), np.zeros(10)])
    error = 2 if cut == "xcut" else 3
    other = 3 if cut == "xcut" else 2
    raw[error] = [0.1, np.inf, np.nan, 0.0, -1.0, -np.inf, 0.2, 0.2, 0.3, 0.4]
    raw[0, 6] = np.nan
    raw[1, 7] = np.inf
    raw[other, 8] = 0.1
    obj.results["edge"][cut] = raw
    obj.sort_fitresults(
        minabserr=floor,
        minrelerr=floor,
        nanbeforemax=False,
        nanopposite=False,
        nanbeforecross=False,
    )
    data = obj._fit_data("edge")
    assert len(data[0]) + len(data[3]) == 2
    obj.write_edgeridge(tmp_path / "valid")
    restored = make_test_pvanalysis()
    restored.read_edgeridge(tmp_path / "valid")
    for actual, expected in zip(restored._fit_data("edge"), data):
        np.testing.assert_equal(actual, expected)
    # If even the remaining measured errors become invalid, no fit is attempted.
    obj.results["edge"][cut][error] = np.inf
    obj.sort_fitresults()
    assert obj.point_status == {"edge": "no_points", "ridge": "no_points"}
    assert obj.fit_edgeridge() == {"edge": None, "ridge": None}
    assert not pv_sampler
    obj.write_edgeridge(tmp_path / "empty")
    restored.read_edgeridge(tmp_path / "empty")
    assert restored.point_status == {"edge": "no_points", "ridge": "no_points"}


def test_pvanalysis_flat_profile_export(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    monkeypatch.setattr(Figure, "savefig", lambda *args, **kwargs: None)
    obj = make_test_pvanalysis()
    obj.fitsdata.data = np.ones_like(obj.fitsdata.data)
    with np.errstate(divide="ignore", invalid="ignore"):
        obj.get_edgeridge(
            str(tmp_path / "flat"),
            quadrant="13",
            Mlim=[],
            use_velocity=False,
            nanbeforemax=False,
            nanopposite=False,
            nanbeforecross=False,
        )
    obj.write_edgeridge(tmp_path / "saved")
    restored = make_test_pvanalysis()
    restored.read_edgeridge(tmp_path / "saved")
    for method in ("edge", "ridge"):
        for actual, expected in zip(restored._fit_data(method), obj._fit_data(method)):
            np.testing.assert_equal(actual, expected)


@pytest.mark.parametrize("cuts", [("xcut",), ("vcut",), ("xcut", "vcut")])
@pytest.mark.parametrize("shift", [-3.0, 1.0, 3.0])
def test_pvanalysis_boundaries_use_fitted_reference(cuts, shift):
    obj = make_test_pvanalysis()
    v0 = np.array([-4.0, -2.0, 2.0, 4.0]) if "xcut" in cuts else np.empty(0)
    x1 = np.array([-50.0, -100.0, 100.0, 50.0]) if "xcut" in cuts else np.empty(0)
    x0 = np.array([-25.0, 150.0]) if "vcut" in cuts else np.empty(0)
    v1 = np.array([-8.0, 4.0 / 3.0]) if "vcut" in cuts else np.empty(0)
    samples = np.array(
        [
            [80.0, 100.0, 120.0],
            [2.0, 3.0, 4.0],
            [0.5, 1.0, 1.5],
            [0.2, 0.4, 0.5],
            [-3.0, -0.7, 1.5],
        ]
    )
    shifted = samples.copy()
    shifted[4] -= shift
    a = obj._range_for(v0, x1, np.ones_like(x1), x0, v1, np.ones_like(v1), samples)
    b = obj._range_for(
        v0 - shift, x1, np.ones_like(x1), x0, v1 - shift, np.ones_like(v1), shifted
    )
    for boundary in range(2):
        np.testing.assert_allclose(a[0][boundary], b[0][boundary])
        # M is proportional to radius times rotation speed squared.
        original_mass = a[0][boundary] * (a[1][boundary] - samples[4]) ** 2
        shifted_mass = b[0][boundary] * (b[1][boundary] - shifted[4]) ** 2
        np.testing.assert_allclose(original_mass, shifted_mass)


def test_pvanalysis_image_only_and_point_plots(tmp_path, monkeypatch):
    from matplotlib.figure import Figure

    saved = []
    monkeypatch.setattr(Figure, "savefig", lambda self, name, **kw: saved.append(self))
    obj = make_test_pvanalysis()
    obj.plot_fitresult(
        outname=tmp_path / "image", show=False, xlim=[5.0, 200.0], vlim=[0.4, 4.0]
    )
    assert len(saved) == 2
    assert all(not fig.axes[0].lines for fig in saved)
    assert obj.point_status == {"edge": "not_run", "ridge": "not_run"}
    obj = load_test_pvanalysis_points(tmp_path, methods=("ridge",))
    saved.clear()
    obj.plot_fitresult(
        outname=tmp_path / "points", show=False, xlim=[5.0, 200.0], vlim=[0.4, 4.0]
    )
    assert len(saved) == 2
    assert all(len(fig.axes[0].containers) == 4 for fig in saved)
    assert obj.popt == {"edge": None, "ridge": None}


@pytest.mark.parametrize("cut", ["xcut", "vcut"])
def test_pvanalysis_linear_fit_is_analytic_and_ridge_only(tmp_path, cut, pv_sampler):
    x = np.array([-100.0, -50.0, 50.0, 100.0])
    rows = np.c_[
        x,
        np.ones(4) if cut == "xcut" else np.zeros(4),
        0.1 + 0.02 * x,
        np.full(4, 0.1) if cut == "vcut" else np.zeros(4),
    ]
    np.savetxt(tmp_path / "line.ridge.dat", rows)
    obj = make_test_pvanalysis()
    obj.read_edgeridge(tmp_path / "line", methods=("ridge",), quadrant="13")
    result = obj.fit_linear()
    np.testing.assert_allclose(result["ridge"]["popt"], [0.1, 0.02])
    assert result["edge"] is None and not pv_sampler


def test_pvanalysis_real_sampler_and_named_corner(tmp_path):
    obj = load_test_pvanalysis_points(tmp_path, methods=("ridge",))
    result = obj.fit_edgeridge(
        methods=("ridge",),
        include_dp=False,
        fixed_pin=1.0,
        outname=tmp_path / "real",
        return_chain=True,
        kwargs_emcee_corner={
            "nwalkers_per_ndim": 8,
            "nburnin": 8,
            "nsteps": 16,
            "range_corner": {"Rb": [20.0, 200.0]},
        },
    )
    assert np.isfinite(result["ridge"]["popt"]).all()
    assert obj.chain["ridge"].shape == (1, 128)
    assert (tmp_path / "real.corner_r.png").stat().st_size > 1000


def test_velgrad_center_extraction():
    vg = VelGrad()
    read_velgrad_test_cube(vg)
    vg.get_2Dcenter(cutoff=5.0, vmask=[-2.0, 2.0], method="mean")

    assert vg.data.shape == (len(vg.v), len(vg.y), len(vg.x))
    assert vg.center["xc"].shape == vg.v.shape
    assert vg.center["yc"].shape == vg.v.shape
    assert np.count_nonzero(np.isfinite(vg.center["xc"])) > 0
    assert np.count_nonzero(np.isfinite(vg.center["yc"])) > 0


def make_synthetic_velgrad():
    vg = VelGrad()
    vg.v = np.arange(-3.0, 4.0)
    vg.x = vg.y = np.linspace(-100.0, 100.0, 5)
    vg.bmaj = 5.0
    x, y = np.zeros(7), 10.0 * vg.v
    x[3] = y[3] = np.nan
    vg.center = {"xc": x, "yc": y, "dxc": np.ones(7), "dyc": np.ones(7)}
    return vg


@pytest.fixture
def vg_sampler(monkeypatch):
    import velgrad._velgrad as module

    calls = []

    def sample(bounds, logprob, **kwargs):
        center = np.mean(bounds, axis=0)
        assert np.isfinite(logprob(center))
        calls.append((bounds.copy(), kwargs))
        result = [center, np.full(len(center), 0.1)]
        if kwargs["return_chain"]:
            result.append(np.tile(center[:, None], (1, 12)))
        if kwargs["return_lnp"]:
            result.append(np.zeros(12))
        return result

    monkeypatch.setattr(module, "emcee_corner", sample)
    return calls


@pytest.mark.parametrize("fixcenter", [False, True])
@pytest.mark.parametrize(
    "chain, lnp", [(False, False), (True, False), (False, True), (True, True)]
)
def test_velgrad_sampler_defaults_and_output_layout(vg_sampler, fixcenter, chain, lnp):
    from velgrad._velgrad import emcee_custom

    bounds = np.array([[-10.0, -10.0, -90.0], [10.0, 10.0, 90.0]])
    result = emcee_custom(
        bounds, lambda p: -np.sum(p**2), fixcenter, return_chain=chain, return_lnp=lnp
    )
    sampled = 1 if fixcenter else 3
    assert len(result) == 2 + chain + lnp
    assert result[0].shape == result[1].shape == (3,)
    if fixcenter:
        np.testing.assert_equal(result[0][:2], [0.0, 0.0])
        np.testing.assert_equal(result[1][:2], [0.0, 0.0])
    if chain:
        assert result[2].shape == (sampled, 12)
    if lnp:
        assert result[3 if chain else 2].shape == (12,)
    used_bounds, options = vg_sampler[0]
    assert used_bounds.shape == (2, sampled)
    assert options["nwalkers_per_ndim"] == 16
    assert options["nburnin"] == 2000 and options["nsteps"] == 2000
    assert options["simpleoutput"] is True


@pytest.mark.parametrize("fixcenter", [False, True])
@pytest.mark.parametrize("lowvelfilter", [False, True])
def test_velgrad_filtering_forwards_all_sampler_options(
    vg_sampler, fixcenter, lowvelfilter
):
    vg = make_synthetic_velgrad()
    labels = ["Angle"] if fixcenter else ["X", "Y", "Angle"]
    settings = {
        "nwalkers_per_ndim": 4,
        "nburnin": 5,
        "nsteps": 6,
        "rangelevel": None,
        "range_corner": {"pa": [-90.0, 90.0], "xoff": [-50.0, 50.0]},
        "labels": labels,
        "show_corner": False,
        "figname": None,
    }
    original = copy.deepcopy(settings)
    result = vg.filtering(
        fixcenter=fixcenter,
        lowvelfilter=lowvelfilter,
        save_points=False,
        print_result=False,
        return_chain=True,
        return_lnp=True,
        kwargs_emcee_corner=settings,
    )
    assert len(vg_sampler) == (5 if lowvelfilter else 1)
    expected_range = (
        [[-90.0, 90.0]] if fixcenter else [[-50.0, 50.0], 0.8, [-90.0, 90.0]]
    )
    for _, options in vg_sampler:
        assert options["nsteps"] == 6 and options["nburnin"] == 5
        assert options["nwalkers_per_ndim"] == 4
        assert options["labels"] == labels
        assert options["range_corner"] == expected_range
        assert options["figname"] is None and options["show_corner"] is False
    assert settings == original
    assert vg.chain_grad.shape == (1 if fixcenter else 3, 12)
    assert vg.lnp_grad.shape == (12,)
    assert result["pa_grad"] == 0.0
    if lowvelfilter:
        assert all(not options["return_chain"] for _, options in vg_sampler[:-1])


@pytest.mark.parametrize("voff_fixed", [0.0, 0.2, -0.2, None])
def test_velgrad_mass_fit_forwards_sampler_options(vg_sampler, voff_fixed, capsys):
    vg = make_synthetic_velgrad()
    radius = np.full(7, np.nan)
    good = vg.v != 0
    radius[good] = np.sign(vg.v[good]) * 100.0 / vg.v[good] ** 2
    vg.kepler = {
        "xc": vg.center["xc"].copy(),
        "yc": radius,
        "dxc": np.ones(7),
        "dyc": np.ones(7),
    }
    vg.pa_grad = 0.0
    settings = {
        "nsteps": 15,
        "nburnin": 5,
        "range_corner": {"M_p": [1.0, 1000.0], "voff": [-0.5, 0.5]},
    }
    original = copy.deepcopy(settings)
    result = vg.calc_mstar(
        voff_fixed=voff_fixed,
        print_result=True,
        return_chain=True,
        return_lnp=True,
        kwargs_emcee_corner=settings,
    )
    expected = ["M_p", "v_break", "p_low"]
    ranges = [[1.0, 1000.0], 0.8, 0.8]
    if voff_fixed is None:
        expected.append("voff")
        ranges.append([-0.5, 0.5])
    options = vg_sampler[0][1]
    assert options["nsteps"] == 15 and options["nburnin"] == 5
    assert options["labels"] == expected
    assert options["range_corner"] == ranges
    assert settings == original
    assert vg.chain_mstar.shape == (len(expected), 12)
    assert vg.lnp_mstar.shape == (12,)
    assert result["popt"].shape == (4,)
    assert np.isfinite(result["Mstar"])

    np.testing.assert_equal(vg.popt, result["popt"])
    np.testing.assert_equal(vg.perr, result["perr"])
    report = capsys.readouterr().out
    if voff_fixed is not None:
        assert result["popt"][-1] == voff_fixed
        assert result["perr"][-1] == 0.0
        assert f"voff = {voff_fixed:.3f} +/- 0.000" in report


@pytest.mark.parametrize("method", ["filtering", "calc_mstar"])
@pytest.mark.parametrize(
    "key",
    ["bounds", "log_prob_fn", "args", "simpleoutput", "return_chain", "return_lnp"],
)
def test_velgrad_rejects_protected_options_before_work(method, key):
    vg = VelGrad()
    with pytest.raises(ValueError, match=key):
        getattr(vg, method)(kwargs_emcee_corner={key: None})


def test_velgrad_real_sampler_options_and_corner(tmp_path):
    vg = make_synthetic_velgrad()
    output = str(tmp_path / "gradient.corner.png")
    vg.filtering(
        save_points=False,
        print_result=False,
        return_chain=True,
        kwargs_emcee_corner={
            "nwalkers_per_ndim": 8,
            "nburnin": 8,
            "nsteps": 16,
            "figname": output,
            "range_corner": {"pa": [-90.0, 90.0]},
        },
    )
    assert vg.chain_grad.shape == (1, 128)
    assert Path(output).stat().st_size > 1000
    result = vg.calc_mstar(
        voff_fixed=None,
        print_result=False,
        return_chain=True,
        return_lnp=True,
        kwargs_emcee_corner={"nwalkers_per_ndim": 8, "nburnin": 8, "nsteps": 16},
    )
    assert vg.chain_mstar.shape == (4, 512)
    assert vg.lnp_mstar.shape == (512,)
    assert np.isfinite(result["Mstar"])


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
