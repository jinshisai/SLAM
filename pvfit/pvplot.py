import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy import constants, units
from scipy.interpolate import RectBivariateSpline as RBS

from pvfit.pvfits import Impvfits



def set_rcparams():
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

       
def nice_ticks(ticks, tlim):
    order = 10**np.floor(np.log10(tlow := tlim[0]))
    tlow = np.ceil(tlow / order) * order
    order = 10**np.floor(np.log10(tup := tlim[1]))
    tup = np.floor(tup / order) * order
    return np.sort(np.r_[ticks, tlow, tup])


def nice_labels(ticks):
    digits = np.floor(np.log10(ticks)).astype('int').clip(None, 0)
    return [f'{t:.{d:d}f}' for t, d in zip(ticks, -digits)]


class PVPlot():
    """make a Position-Velocity diagram in color and/or contour maps.
    
    """
    def __init__(self, fig=None, ax=None, fitsimage: str = None,
                 restfrq: float = None, beam: list = None,
                 pa: float = None, vsys: float = 0,
                 dist: float = 1.,
                 d: list = None, x: list = None, v: list = None,
                 xlim: list = [1, 1e10], vlim: list = [0.1, 1e10],
                 flipaxis: bool = False, quadrant: str = None,
                 loglog: bool = False):
        set_rcparams()
        if fig is None:
            fig = plt.figure(figsize=(7, 5))
        if ax is None:
            ax = fig.add_subplot(1, 1, 1)
        self.fig, self.ax = fig, ax
        if fitsimage is not None:
            fitsdata = Impvfits(fitsimage, pa=pa)
            d = fitsdata.data
            x, v = fitsdata.xaxis * dist, fitsdata.vaxis
            if 'BUNIT' in (h := fitsdata.header):
                self.bunit = h['BUNIT']
            restfrq = fitsdata.restfreq
            beam = fitsdata.beam
        d = np.squeeze(d)
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
        self.d, self.x, self.v = d, x, v
        self.restfrq = restfrq
        self.beam = beam
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
        
    def add_color(self, restfrq: float = None, bmaj: float = None,
                  bmin: float = None, bpa: float = None,
                  Tb: bool = False, log: bool = False,
                  show_cbar: bool = True,
                  cblabel: str = None, cbformat: float = '%.1e',
                  cbticks: list = None, cbticklabels: list = None,
                  **kwargs) -> None:
        kwargs0 = {'cmap': 'viridis', 'zorder': 1}
        if restfrq is None:
            restfrq = self.restfrq
        if bmaj is None or bmin is None or bpa is None:
            bmaj, bmin, bpa = self.beam
        if self.loglog:
            self.gen_loglog()
            x, v, d = self.xl, self.vl, self.dl
        else:
            x, v, d = self.x, self.v, self.d
        if Tb:
            Omega = bmaj * bmin / 3600.**2 * np.pi / 4. / np.log(2.)
            lam = constants.c.to('m/s').value / restfrq
            Jy2K = units.Jy.to('J*s**(-1)*m**(-2)*Hz**(-1)') \
                   * lam**2 / 2. / constants.k_B.to('J/K').value / Omega
            d *= Jy2K
        if log:
            vmin = kwargs['vmin'] if 'vmin' in kwargs else np.nanstd(d)
            kwargs['vmin'] = np.log10(vmin)
            vmax = kwargs['vmax'] if 'vmax' in kwargs else np.nanmax(d)
            kwargs['vmax'] = np.log10(vmax)
        kwargs0 = dict(kwargs0, **kwargs)
        ax = self.ax
        if self.flipaxis: x, v, d = v, x, d.T
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
                cb.set_ticks(t := cb.get_ticks())
                cb.set_ticklabels([f'{lt:.1e}' for lt in 10**t])


    def add_contour(self, restfrq: float = None, bmaj: float = None,
                    bmin: float = None, bpa: float = None,
                    Tb: bool = False, rms: float = None,
                    levels: list = [3, 6],
                    **kwargs) -> None:
        kwargs0 = {'colors': 'lime', 'linewidths': 1.2, 'zorder': 2}
        if restfrq is None:
            restfrq = self.restfrq
        if None in [bmaj, bmin, bpa]:
            bmaj, bmin, bpa = self.beam
        if self.loglog:
            self.gen_loglog()
            x, v, d = self.xl, self.vl, self.dl
        else:
            x, v, d = self.x, self.v, self.d
        if Tb:
            Omega = bmaj * bmin / 3600.**2 * np.pi / 4. / np.log(2.)
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
        if self.flipaxis: x, v, d = v, x, d.T
        ax.contour(x, v, d, np.array(levels) * rms, **kwargs0)
        if self.loglog:
            ax.set_xscale('log')
            ax.set_yscale('log')
 
                 
    def set_axis(self, xticks: list = None, yticks: list = None,
                 xticklabels: list = None, yticklabels: list = None,
                 xlabel: str = None, ylabel: str = None) -> None:
        (xmin, xmax), (vmin, vmax) = self.xlim, self.vlim
        rlabel, vlabel = 'Radius (au)', r'Velocity (km s$^{-1}$)'
        ax = self.ax
        if self.loglog:
            ax.set_xticks(xticks := nice_ticks(ax.get_xticks(), (xmin, xmax)))
            ax.set_yticks(yticks := nice_ticks(ax.get_yticks(), (vmin, vmax)))
            ax.set_xticklabels(nice_labels(xticks))
            ax.set_yticklabels(nice_labels(yticks))
            ax.set_aspect(1)
            ax.set_xlim(xmin * 0.999, xmax * 1.001)
            ax.set_ylim(vmin * 0.999, vmax * 1.001)
        else:
            if self.flipaxis:
                x0, x1, y0, y1 = -vmax, vmax, -xmax, xmax
                rlabel, vlabel = vlabel, rlabel
            else:
                x0, x1, y0, y1 = -xmax, xmax, -vmax, vmax
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
        ax.set_xlabel(rlabel if xlabel is None else xlabel)
        ax.set_ylabel(vlabel if ylabel is None else ylabel)
        if xticklabels: ax.set_xticklabels(xticklabels)
        if yticklabels: ax.set_yticklabels(yticklabels)
        self.fig.tight_layout()
            
                
    def savefig(self, figname: str = None, show: bool = False,
                **kwargs) -> None:
        kwargs0 = {'bbox_inches': 'tight', 'transparent': True}
        if figname is not None:
            self.fig.savefig(figname, **dict(kwargs0, **kwargs))
        if show: plt.show()
        plt.close()
        