import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from astropy import constants, units
from scipy.interpolate import RectBivariateSpline as RBS
from typing import Any

from pvanalysis.pvfits import Impvfits


def set_rcparams() -> None:
    """Set the Matplotlib defaults used by PV-analysis figures."""
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.linewidth'] = 1.5
    plt.rcParams['xtick.direction'] = 'inout'
    plt.rcParams['ytick.direction'] = 'inout'
    plt.rcParams['xtick.major.size'] = 12
    plt.rcParams['ytick.major.size'] = 12
    plt.rcParams['xtick.minor.size'] = 8
    plt.rcParams['ytick.minor.size'] = 8
    plt.rcParams['xtick.major.width'] = 1.5
    plt.rcParams['ytick.major.width'] = 1.5
    plt.rcParams['xtick.minor.width'] = 1.5
    plt.rcParams['ytick.minor.width'] = 1.5


def nice_ticks(ticks: np.ndarray | list[float],
               tlim: tuple[float, float] | list[float]) -> np.ndarray:
    """Add rounded lower and upper limits to logarithmic tick locations.

    Args:
        ticks (np.ndarray or list): Existing positive tick locations.
        tlim (tuple or list): Positive lower and upper axis limits.

    Returns:
        np.ndarray: Sorted ticks including rounded values near both limits.
    """
    order = 10**np.floor(np.log10(tlow := tlim[0]))
    tlow = np.ceil(tlow / order) * order
    order = 10**np.floor(np.log10(tup := tlim[1]))
    tup = np.floor(tup / order) * order
    return np.sort(np.r_[ticks, tlow, tup])


def nice_labels(ticks: np.ndarray | list[float]) -> list[str]:
    """Format numeric tick locations with scale-dependent precision.

    Args:
        ticks (np.ndarray or list): Positive tick locations.

    Returns:
        list: Formatted tick labels without unnecessary decimal places.
    """
    digits = np.floor(np.log10(ticks)).astype('int').clip(None, 0)
    return [f'{t:.{d}f}' for t, d in zip(ticks, -digits)]


class PVPlot():
    """Make a position-velocity diagram with color and contour maps.

    Args:
        fig (matplotlib.figure.Figure or None, optional): Existing figure.
            None creates a new figure. Defaults to None.
        ax (matplotlib.axes.Axes or None, optional): Existing axes. None adds
            axes to ``fig``. Defaults to None.
        fitsimage (str or None, optional): Input PV FITS file. When supplied,
            its data, axes, frequency, and beam override the corresponding
            array arguments. Defaults to None.
        restfrq (float or None, optional): Rest frequency in Hz, required for
            brightness-temperature conversion. Defaults to None.
        beam (np.ndarray, list, or None, optional): Beam
            ``[major, minor, PA]`` or a per-channel beam table. Major and minor
            axes are in arcseconds and PA is in degrees. Defaults to None.
        pa (float or None, optional): Position angle of the PV cut in degrees,
            used when reading ``fitsimage``. Defaults to None.
        multibeam (bool, optional): Whether ``beam`` is a per-channel beam
            table. Defaults to False.
        vsys (float, optional): Systemic velocity subtracted from the velocity
            axis in km/s. Defaults to 0.
        dist (float, optional): Source distance in pc, used to convert spatial
            offsets from arcseconds to au. Defaults to 1.
        d (np.ndarray or None, optional): Two-dimensional PV intensity array.
            Required when ``fitsimage`` is None. Defaults to None.
        x (np.ndarray or None, optional): Spatial-offset axis in arcseconds.
            Required when ``fitsimage`` is None. Defaults to None.
        v (np.ndarray or None, optional): Velocity axis in km/s. Required when
            ``fitsimage`` is None. Defaults to None.
        xlim (list, optional): Minimum and maximum absolute plotted positions
            in au. Defaults to [1e-10, 1e10].
        vlim (list, optional): Minimum and maximum absolute plotted velocities
            relative to ``vsys`` in km/s. Defaults to [1e-10, 1e10].
        flipaxis (bool, optional): Whether velocity is horizontal and position
            is vertical. Defaults to False.
        quadrant (str or None, optional): Bright pair of PV quadrants,
            ``'13'`` or ``'24'``. None determines it from the data. Defaults
            to None.
        loglog (bool, optional): Whether to fold the selected opposite
            quadrants into positive position-velocity space for a log-log
            plot. Defaults to False.
    """
    def __init__(self, fig: Figure | None = None, ax: Axes | None = None,
                 fitsimage: str | None = None,
                 restfrq: float | None = None,
                 beam: np.ndarray | list[float] | None = None,
                 pa: float | None = None, multibeam: bool = False,
                 vsys: float = 0, dist: float = 1.,
                 d: np.ndarray | None = None,
                 x: np.ndarray | None = None,
                 v: np.ndarray | None = None,
                 xlim: list[float] = [1e-10, 1e10],
                 vlim: list[float] = [1e-10, 1e10],
                 flipaxis: bool = False, quadrant: str | None = None,
                 loglog: bool = False) -> None:
        """Initialize a PV figure from a FITS file or supplied arrays."""
        set_rcparams()
        if fig is None:
            fig = plt.figure(figsize=(7, 5))
        if ax is None:
            ax = fig.add_subplot(1, 1, 1)
        self.fig, self.ax = fig, ax
        if fitsimage is not None:
            fitsdata = Impvfits(fitsimage, pa=pa, multibeam=multibeam)
            d = fitsdata.data
            x, v = fitsdata.xaxis, fitsdata.vaxis
            if 'BUNIT' in (h := fitsdata.header):
                self.bunit = h['BUNIT']
            restfrq = fitsdata.restfreq
            beam = fitsdata.beam
            multibeam = fitsdata.multibeam
        d = np.squeeze(d)
        x = x * dist
        v = v - vsys
        xlim[0] = max(xlim[0], np.abs(x[1] - x[0]))
        xlim[1] = min(xlim[1], -x[0], x[-1])
        vlim[0] = max(vlim[0], np.abs(v[1] - v[0]))
        vlim[1] = min(vlim[1], -v[0], v[-1])
        i0 = np.argmin(np.abs(x + xlim[1]))
        i1 = np.argmin(np.abs(x - xlim[1])) + 1
        j0 = np.argmin(np.abs(v + vlim[1]))
        j1 = np.argmin(np.abs(v - vlim[1])) + 1
        d, x, v = d[j0:j1, i0:i1], x[i0:i1], v[j0:j1]
        self.jrange = [j0, j1]
        self.d, self.x, self.v = d, x, v
        self.restfrq = restfrq
        self.beam = beam
        self.multibeam = multibeam
        self.flipaxis = flipaxis
        self.loglog = loglog
        self.xlim = xlim
        self.vlim = vlim
        if quadrant is None:
            ic = np.argmin(np.abs(x))
            jc = np.argmin(np.abs(v))
            q = np.mean(d[:jc, :ic]) + np.mean(d[jc:, ic:]) \
                - np.mean(d[:jc, ic:]) - np.mean(d[jc:, :ic])
            self.q13 = (q > 0)
        else:
            self.q13 = (quadrant == '13')

    def gen_loglog(self) -> None:
        """Fold opposite PV quadrants onto a positive log-log grid.

        The interpolated position, velocity, and intensity arrays are stored
        as ``self.xl``, ``self.vl``, and ``self.dl``.
        """
        dx, dv = self.x[1] - self.x[0], self.v[1] - self.v[0]
        mi = int(self.x[-1] - self.x[0] / dx)
        ni = int(self.v[-1] - self.v[0] / dv)
        xi = np.linspace(-mi * dx, mi * dx, 2 * mi + 1)
        vi = np.linspace(-ni * dv, ni * dv, 2 * ni + 1)
        d = self.d if self.q13 else self.d[:, ::-1]
        di = RBS(self.v, self.x, d)(vi, xi)
        d = (di + di[::-1, ::-1]) / 2.
        i0 = np.argmin(np.abs(xi - self.xlim[0]))
        j0 = np.argmin(np.abs(vi - self.vlim[0]))
        xi, vi, d = xi[i0:], vi[j0:], d[j0:, i0:]
        self.xl, self.vl, self.dl = xi, vi, d

    def add_color(self, restfrq: float | None = None,
                  bmaj: float | np.ndarray | None = None,
                  bmin: float | np.ndarray | None = None,
                  bpa: float | np.ndarray | None = None,
                  Tb: bool = False, log: bool = False,
                  show_cbar: bool = True,
                  cblabel: str | None = None, cbformat: str = '%.1e',
                  cbticks: list[float] | np.ndarray | None = None,
                  cbticklabels: list[str] | None = None,
                  **kwargs: Any) -> None:
        """Add the PV intensity as a color map.

        Args:
            restfrq (float or None, optional): Rest frequency in Hz. None uses
                the value stored by the constructor. Defaults to None.
            bmaj (float, np.ndarray, or None, optional): Beam major axis in
                arcseconds. An array supplies one value per channel. Defaults
                to None.
            bmin (float, np.ndarray, or None, optional): Beam minor axis in
                arcseconds. An array supplies one value per channel. Defaults
                to None.
            bpa (float, np.ndarray, or None, optional): Beam position angle in
                degrees. An array supplies one value per channel. Defaults to
                None.
            Tb (bool, optional): Whether to convert Jy/beam intensity to
                brightness temperature. Defaults to False.
            log (bool, optional): Whether to plot the base-10 logarithm of
                positive intensity. Defaults to False.
            show_cbar (bool, optional): Whether to add a color bar. Defaults to
                True.
            cblabel (str or None, optional): Color-bar label. None uses the
                FITS brightness unit when available. Defaults to None.
            cbformat (str, optional): Matplotlib color-bar number format.
                Defaults to ``'%.1e'``.
            cbticks (list, np.ndarray, or None, optional): Color-bar tick
                values in linear intensity units. Defaults to None.
            cbticklabels (list or None, optional): Explicit color-bar tick
                labels. Defaults to None.
            **kwargs: Additional arguments passed to ``Axes.pcolormesh``.
        """
        kwargs0 = {'cmap': 'viridis', 'zorder': 1, 'shading': 'nearest'}
        if restfrq is None:
            restfrq = self.restfrq
        if bmaj is None or bmin is None or bpa is None:
            if self.multibeam:
                bmaj = self.beam['BMAJ']
                bmin = self.beam['BMIN']
                bpa = self.beam['BPA']
                if self.loglog:
                    ichan = np.nanargmax(bmaj * bmin)
                    bmaj, bmin, bpa = bmaj[ichan], bmin[ichan], bpa[ichan]
            else:
                bmaj, bmin, bpa = self.beam
        if self.loglog:
            self.gen_loglog()
            x, v, d = self.xl, self.vl, self.dl
        else:
            x, v, d = self.x, self.v, self.d
        if Tb:
            Omega = bmaj * bmin / 3600.**2 * np.radians(1)**2 \
                * np.pi / 4. / np.log(2.)
            if isinstance(Omega, np.ndarray):
                j0, j1 = self.jrange
                Omega = np.tile(Omega[j0:j1], (len(x), 1)).T
            lam = constants.c.to('m/s').value / restfrq
            Jy2K = units.Jy.to('J*s**(-1)*m**(-2)*Hz**(-1)') \
                * lam**2 / 2. / constants.k_B.to('J/K').value / Omega
            d = d * Jy2K
        if log:
            vmin = kwargs['vmin'] if 'vmin' in kwargs else np.nanstd(d)
            kwargs['vmin'] = np.log10(vmin)
            vmax = kwargs['vmax'] if 'vmax' in kwargs else np.nanmax(d)
            kwargs['vmax'] = np.log10(vmax)
        if log:
            d = np.log10(d.clip(np.min(d[d > 0]), None))
        kwargs0 = dict(kwargs0, **kwargs)
        ax = self.ax
        if self.flipaxis:
            x, v, d = v, x, d.T
        p = ax.pcolormesh(x, v, d, **kwargs0)
        if self.loglog:
            ax.set_xscale('log')
            ax.set_yscale('log')
        if show_cbar:
            if cblabel is None and hasattr(self, 'bunit'):
                cblabel = self.bunit
            cb = plt.colorbar(p, ax=self.ax, label=cblabel, format=cbformat)
            cb.ax.tick_params(labelsize=14)
            font = mpl.font_manager.FontProperties(size=16)
            cb.ax.yaxis.label.set_font_properties(font)
            if cbticks is not None:
                cb.set_ticks(np.log10(cbticks) if log else cbticks)
            if cbticklabels is not None:
                cb.set_ticklables(cbticklabels)
            elif log:
                t = cb.get_ticks()
                t = t[(kwargs['vmin'] < t) * (t < kwargs['vmax'])]
                cb.set_ticks(t)
                cb.set_ticklabels([f'{lt:.1e}' for lt in 10**t])

    def add_contour(self, restfrq: float | None = None,
                    bmaj: float | np.ndarray | None = None,
                    bmin: float | np.ndarray | None = None,
                    bpa: float | np.ndarray | None = None,
                    Tb: bool = False, rms: float | None = None,
                    levels: list[float] | np.ndarray = [3, 6],
                    **kwargs: Any) -> None:
        """Add PV intensity contours.

        Args:
            restfrq (float or None, optional): Rest frequency in Hz. None uses
                the value stored by the constructor. Defaults to None.
            bmaj (float, np.ndarray, or None, optional): Beam major axis in
                arcseconds. An array supplies one value per channel. Defaults
                to None.
            bmin (float, np.ndarray, or None, optional): Beam minor axis in
                arcseconds. An array supplies one value per channel. Defaults
                to None.
            bpa (float, np.ndarray, or None, optional): Beam position angle in
                degrees. An array supplies one value per channel. Defaults to
                None.
            Tb (bool, optional): Whether to convert Jy/beam intensity to
                brightness temperature. Defaults to False.
            rms (float or None, optional): RMS intensity used to scale
                ``levels``. None estimates it from the data edges. Defaults to
                None.
            levels (list or np.ndarray, optional): Contour levels in units of
                ``rms``. Defaults to [3, 6].
            **kwargs: Additional arguments passed to ``Axes.contour``.
        """
        kwargs0 = {'colors': 'lime', 'linewidths': 1.2, 'zorder': 2}
        if restfrq is None:
            restfrq = self.restfrq
        if None in [bmaj, bmin, bpa]:
            if self.multibeam:
                bmaj = self.beam['BMAJ']
                bmin = self.beam['BMIN']
                bpa = self.beam['BPA']
                if self.loglog:
                    ichan = np.nanargmax(bmaj * bmin)
                    bmaj, bmin, bpa = bmaj[ichan], bmin[ichan], bpa[ichan]
            else:
                bmaj, bmin, bpa = self.beam
        if self.loglog:
            self.gen_loglog()
            x, v, d = self.xl, self.vl, self.dl
        else:
            x, v, d = self.x, self.v, self.d
        if Tb:
            Omega = bmaj * bmin / 3600.**2 * np.pi / 4. / np.log(2.)
            if isinstance(Omega, np.ndarray):
                j0, j1 = self.jrange
                Omega = np.tile(Omega[j0:j1], (len(x), 1)).T
            lam = constants.c.to('m/s').value / restfrq
            Jy2K = units.Jy.to('J*s**(-1)*m**(-2)*Hz**(-1)') \
                * lam**2 / 2. / constants.k_B.to('J/K').value / Omega
            d *= Jy2K
        if rms is None:
            rms = (np.std(d[:5, :]) + np.std(d[-5:, :])
                   + np.std(d[:, :5]) + np.std(d[:, -5:])) / 4.
            print(f'rms = {rms:.2e}')
        kwargs0 = dict(kwargs0, **kwargs)
        ax = self.ax
        if self.flipaxis:
            x, v, d = v, x, d.T
        ax.contour(x, v, d, np.array(levels) * rms, **kwargs0)
        if self.loglog:
            ax.set_xscale('log')
            ax.set_yscale('log')

    def set_axis(self, xticks: list[float] | None = None,
                 yticks: list[float] | None = None,
                 xticklabels: list[str] | None = None,
                 yticklabels: list[str] | None = None,
                 xlabel: str | None = None,
                 ylabel: str | None = None) -> None:
        """Set PV axis limits, scales, labels, and tick labels.

        Args:
            xticks (list or None, optional): Reserved for custom horizontal
                ticks. The current implementation determines ticks from the
                axes. Defaults to None.
            yticks (list or None, optional): Reserved for custom vertical
                ticks. The current implementation determines ticks from the
                axes. Defaults to None.
            xticklabels (list or None, optional): Custom horizontal tick
                labels. Defaults to None.
            yticklabels (list or None, optional): Custom vertical tick labels.
                Defaults to None.
            xlabel (str or None, optional): Custom horizontal-axis label. None
                selects a position or velocity label. Defaults to None.
            ylabel (str or None, optional): Custom vertical-axis label. None
                selects a position or velocity label. Defaults to None.
        """
        (xmin, xmax), (vmin, vmax) = self.xlim, self.vlim
        rlabel, vlabel = 'Radius (au)', r'Velocity (km s$^{-1}$)'
        ax = self.ax
        if self.flipaxis:
            xmin, xmax, vmin, vmax = vmin, vmax, xmin, xmax
            rlabel, vlabel = vlabel, rlabel
        if self.loglog:
            ax.set_xticks(xticks := nice_ticks(ax.get_xticks(), (xmin, xmax)))
            ax.set_yticks(yticks := nice_ticks(ax.get_yticks(), (vmin, vmax)))
            ax.set_xticklabels(nice_labels(xticks))
            ax.set_yticklabels(nice_labels(yticks))
            ax.set_aspect(1)
            ax.set_xlim(xmin * 0.999, xmax * 1.001)
            ax.set_ylim(vmin * 0.999, vmax * 1.001)
        else:
            ax.set_xlim(-xmax, xmax)
            ax.set_ylim(-vmax, vmax)
        ax.set_xlabel(rlabel if xlabel is None else xlabel)
        ax.set_ylabel(vlabel if ylabel is None else ylabel)
        if xticklabels:
            ax.set_xticklabels(xticklabels)
        if yticklabels:
            ax.set_yticklabels(yticklabels)
        self.fig.tight_layout()

    def savefig(self, figname: str | None = None, show: bool = False,
                **kwargs: Any) -> None:
        """Save, optionally show, and close the PV figure.

        Args:
            figname (str or None, optional): Output figure name. None skips
                saving. Defaults to None.
            show (bool, optional): Whether to show the figure before closing
                it. Defaults to False.
            **kwargs: Additional arguments passed to ``Figure.savefig``.
        """
        kwargs0 = {'bbox_inches': 'tight', 'transparent': True}
        if figname is not None:
            self.fig.savefig(figname, **dict(kwargs0, **kwargs))
        if show:
            plt.show()
        plt.close()
