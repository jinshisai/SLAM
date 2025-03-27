# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Yusuke Aso
# Created Date: 2022 Jan 27
# version = alpha
# ---------------------------------------------------------------------------
"""
This script derives the 2D central position at each velocity channel from a channel map, in the FITS form (AXIS1=deg, AXIS2=deg, AXIS3=Hz) and fits the major-offset vs. velocity with a power-law function. The outputs are the central points on the R.A.-Dec., major-minor, and major-velocity planes.
The main class ChannelAnalysis can be imported to do each steps separately: get the central points, write them, fit them, output the fit result, and plot the central points.

Note. FITS files with multiple beams are not supported. The dynamic range for xlim_plot and vlim_plot should be >10 for nice tick labels.
"""


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy import constants, units
from scipy.optimize import curve_fit

from utils import emcee_corner, ReadFits, rot


GG = constants.G.si.value
M_sun = constants.M_sun.si.value
au = units.au.to('m')
unit = 1.e6 * au / GG / M_sun

def gauss2d(xy, peak, cx, cy, wx, wy, pa):
    x, y = xy
    s, t = rot(x - cx, y - cy, pa)
    return np.ravel(peak * np.exp2(-s**2 / wx**2 - t**2 / wy**2))

def emcee_custom(plim, lnprob, fixcenter):
    popt, perr = emcee_corner(plim[:, -1:] if fixcenter else plim,
                              lnprob,
                              nwalkers_per_ndim=8,
                              nburnin=1000, nsteps=4000,
                              simpleoutput=True)
    if fixcenter:
       popt = np.array([0, 0, popt[0]])
       perr = np.array([0, 0, perr[0]])
    return popt, perr

def r_kep_out(v, M_p, v_break, p_low, vsys):
    v_s, v_a = np.sign(v - vsys), np.abs(v - vsys)
    p = 2. + (p_low - 2.) * (1 + np.sign(v_break - v_a)) / 2.
    r_break = M_p / v_break**2
    return v_s * r_break * (v_a / v_break)**(-p)


class TwoDGrad(ReadFits):

    #def __init__(self):
    
    def get_2Dcenter(self, cutoff: float = 5, vmask: list = [0, 0],
                     minrelerr: float = 0.01, minabserr: float = 0.1,
                     method: str = 'mean'):
        dx, dy = self.dx, self.dy
        xmax, ymax = np.max(self.x), np.max(self.y)
        sigma, data = self.sigma, self.data

        def clipped_error(err, val):
            return max(err, minrelerr * np.abs(val), minabserr * self.bmaj)

        X_org, Y_org = np.meshgrid(self.x, self.y)
        xc, yc, dxc, dyc = [], [], [], []
        for d_org, v in zip(data, self.v):
            cond = d_org > cutoff * sigma
            d, X, Y = d_org[cond], X_org[cond], Y_org[cond]
            xval, xerr, yval, yerr = np.nan, np.nan, np.nan, np.nan
            if len(d) > 1 and (v < vmask[0] or vmask[1] < v):
                if method == 'mean':
                    xval = np.sum(d * X) / np.sum(d)
                    xerr = sigma * np.sqrt(np.sum((X - xval)**2)) / np.sum(d)
                    yval = np.sum(d * Y) / np.sum(d)
                    yerr = sigma * np.sqrt(np.sum((Y - yval)**2)) / np.sum(d)
                elif method == 'peak':
                    xval = X[np.argmax(d)]
                    xerr = self.bmaj / (np.max(d) / sigma)
                    yval = Y[np.argmax(d)]
                    yerr = self.bmaj / (np.max(d) / sigma)
                elif method == 'gauss':
                    if len(d) < 7: continue
                    bounds = [[0, -xmax, -ymax, dx, dy, 0],
                              [d.max() * 2, xmax, ymax, xmax, ymax, np.pi]]
                    try:
                        popt, pcov = curve_fit(gauss2d,
                                               (X.ravel(), Y.ravel()),
                                               d.ravel(), max_nfev=1000,
                                               sigma=X.ravel() * 0 + sigma,
                                               absolute_sigma=True,
                                               bounds=bounds)
                        xval, yval = popt[[1, 2]]
                        xerr, yerr = np.sqrt(np.diag(pcov))[[1, 2]]
                    except RuntimeError:
                        pass
                xerr = clipped_error(xerr, xval)
                yerr = clipped_error(yerr, yval)
            xc.append(xval)
            dxc.append(xerr)
            yc.append(yval)
            dyc.append(yerr)
        xc, dxc, yc, dyc = np.array([xc, dxc, yc, dyc])
        self.center = {'xc':xc, 'dxc':dxc, 'yc':yc, 'dyc':dyc}
        
    def filtering(self, pa0: float = 0.0, fixcenter: bool = False,
                  axisfilter: bool = True, lowvelfilter: bool = True):
        xc = self.center['xc'] * 1
        yc = self.center['yc'] * 1
        dxc = self.center['dxc'] * 1
        dyc = self.center['dyc'] * 1
        n = len(self.v)
        n0 = np.argmin(np.abs(self.v))
        if not fixcenter:
            for i in range(n):
                j = 2 * n0 - i
                if 0 < j or j <= n:
                    xc[i] = yc[i] = dxc[i] = dyc[i] = np.nan
                elif np.isnan(xc[i]) or np.isnan(yc[i]):
                    xc[i] = yc[i] = dxc[i] = dyc[i] = np.nan
                    xc[j] = yc[j] = dxc[j] = dyc[j] = np.nan
        if not np.any(~np.isnan(xc) * ~np.isnan(yc)):
                print('No blue-red pair.')
        
        def bad_channels(x_in, y_in, xoff, yoff, pa):
            if np.all(np.isnan(x_in) | np.isnan(y_in)):
                return np.full_like(x_in, False)
            x0 = x_in - xoff
            y0 = y_in - yoff
            if fixcenter:
                x = x0 * 0
                y = y0 * 0
            else:
                x = x0 + x0[::-1]
                y = y0 + y0[::-1]
            parad = np.radians(pa)
            d = x0 * np.cos(parad) - y0 * np.sin(parad)
            if not axisfilter:
                d = d * 0
            sx2 = np.clip(np.nanmean(x**2), 1e-10, None)
            sy2 = np.clip(np.nanmean(y**2), 1e-10, None)
            sd2 = np.clip(np.nanmean(d**2), 1e-10, None)
            a = x**2 / sx2 + y**2 / sy2 + d**2 / sd2
            if fixcenter:
                return a > 3.84  # 3.84, 6.65, 8.81 covers 95, 99, 99.7%
            else:
                return a > 7.82  # 7.82, 11.35, 13.94 covers 95, 99, 99.7%

        def chi2(p, x_in, y_in, dx_in, dy_in):
            xoff, yoff, pa = p
            c = ~np.isnan(x_in) & ~np.isnan(y_in)
            x, y, dx, dy = x_in[c], y_in[c], dx_in[c], dy_in[c]
            x = (x - xoff) / dx
            y = (y - yoff) / dy
            if fixcenter:
                d1 = x * 0
                d2 = y * 0
            else:
                d1 = (x + x[::-1])**2
                d2 = (y + y[::-1])**2
            parad = np.radians(pa)
            d3 = (x * np.cos(parad) - y * np.sin(parad))**2
            d3 = d3 / ((dx**2 + dy**2) / 2)
            return np.sum(d1 + d2 + d3)
            
        def low_velocity(x_in, y_in, pa_in):
            parad = np.radians(pa_in)
            cospa = np.cos(parad)
            sinpa = np.sin(parad)
            c = np.full_like(x_in, False)
            if np.all(np.isnan(x_in) | np.isnan(y_in)):
                return c
            x = x_in[:n0]
            y = y_in[:n0]
            if np.any(~np.isnan(x)) and np.any(~np.isnan(y)):
                imax = np.nanargmax(np.abs(x * sinpa + y * cospa))
                c[imax+1:n0+1] = True
            x = x_in[n0+1:]
            y = y_in[n0+1:]
            if np.any(~np.isnan(x)) and np.any(~np.isnan(y)):
                imax = np.nanargmax(np.abs(x * sinpa + y * cospa))
                c[n0:n0+1 + imax] = True
            return c.astype('bool')

        goodsolution = False
        xoff = yoff = pa_grad = np.nan
        while not goodsolution:
            if np.all(np.isnan(xc) | np.isnan(yc)):
                print('No point survived.')
                break
            plim = np.array([[self.x.min(), self.y.min(), pa0 - 90.0],
                             [self.x.max(), self.y.max(), pa0 + 90.0]])
            if lowvelfilter:
                c1 = np.full_like(xc, False).astype(bool)
                for _ in range(4):
                    args = np.array([xc, yc, dxc, dyc]) * 1
                    args[0][c1] = args[1][c1] = args[2][c1] = args[3][c1] = np.nan
                    if fixcenter:
                        lnprob = lambda p: -0.5 * chi2([0, 0, p], *args)
                    else:
                        lnprob = lambda p: -0.5 * chi2(p, *args)
                    popt, perr = emcee_custom(plim, lnprob, fixcenter)
                    xoff, yoff, pa_grad = popt
                    print('xoff, yoff, pa ='
                          + f' {popt[0]:.2f}+/-{perr[0]:.2f} au,'
                          + f' {popt[1]:.2f}+/-{perr[1]:.2f} au,'
                          + f' {popt[2]:.2f}+/-{perr[2]:.2f} deg')
                    c1 = low_velocity(args[0] - xoff, args[1] - yoff, pa_grad)
                xc[c1] = yc[c1] = dxc[c1] = dyc[c1] = np.nan
            args = np.array([xc, yc, dxc, dyc])
            if fixcenter:
                lnprob = lambda p: -0.5 * chi2([0, 0, p], *args)
            else:
                lnprob = lambda p: -0.5 * chi2(p, *args)
            popt, perr = emcee_custom(plim, lnprob, fixcenter)
            xoff, yoff, pa_grad = popt
            print('xoff, yoff, pa ='
                  + f' {popt[0]:.2f}+/-{perr[0]:.2f} au,'
                  + f' {popt[1]:.2f}+/-{perr[1]:.2f} au,'
                  + f' {popt[2]:.2f}+/-{perr[2]:.2f} deg')
            c2 = bad_channels(xc, yc, xoff, yoff, pa_grad)
            if np.any(c2):
                xc[c2] = yc[c2] = dxc[c2] = dyc[c2] = np.nan
            else:
                goodsolution = True
        
        xc, yc = xc - xoff, yc - yoff
        self.xoff, self.yoff, self.pa_grad = popt
        self.dxoff, self.dyoff, self.dpa_grad = perr
        self.kepler = {'xc':xc, 'dxc':dxc, 'yc':yc, 'dyc':dyc}
        dof = len(xc[~np.isnan(xc)]) - (1. if fixcenter else 3.) - 1
        self.chi2r_grad = chi2(popt, xc, yc, dxc, dyc) / dof
        
    def calc_mstar(self, incl: float = 90,
                   voff_range: list = [-0.5, 0.5],
                   voff_fixed: float | None = None,
                   minabserr: float = 0.1, minrelerr: float = 0.01):
        self.incl = incl
        sini2 = np.sin(np.radians(incl))**2
        xc = self.kepler['xc'] * 1
        yc = self.kepler['yc'] * 1
        dxc = self.kepler['dxc'] * 1
        dyc = self.kepler['dyc'] * 1
        self.Rkep = np.nan
        self.Vkep = np.nan
        self.vmid = np.nan
        self.Mstar = np.nan
        self.popt = None
        if not np.any(c := ~np.isnan(xc) * ~np.isnan(yc)):
            print('No point to calculate Rkep, Vkep, and Mstar.')
        else:
            v, x, y, dx, dy = self.v[c], xc[c], yc[c], dxc[c], dyc[c]
            sin_g = np.sin(np.radians(self.pa_grad))
            cos_g = np.cos(np.radians(self.pa_grad))
            r = x * sin_g + y * cos_g
            dr = np.hypot(dx * sin_g, dy * cos_g)
            dr = np.max([dr, minrelerr * np.abs(r),
                         [minabserr * self.bmaj] * len(dr)], axis=0)
            s_model = np.sign(np.sum(r * v))
            Rkep = np.max(np.abs(r)) / 0.760  # Appendix A in Aso+15_ApJ_812_27
            Vkep = np.min(np.abs(v))
            print(f'Max r = {Rkep:.1f} au at v = {Vkep:.2f} km/s'
                  + ' (1/0.76 corrected)')
            self.Rkep = Rkep
            self.Vkep = Vkep
            def lnprob(p):
                if voff_fixed is None:
                    M_p, v_break, p_low, vsys = p
                else:
                    M_p, v_break, p_low = p
                    vsys = voff_fixed
                r_model =  r_kep_out(v, M_p, v_break, p_low, vsys)
                chi2 = np.sum(((r - s_model * r_model) / dr)**2)
                return -0.5 * chi2
            Mmin = np.min(np.abs(r)) * np.min(np.abs(v))**2
            Mmax = np.max(np.abs(r)) * np.max(np.abs(v))**2
            plim = np.array([[Mmin, np.min(np.abs(v)), -10, voff_range[0]],
                             [Mmax, np.max(np.abs(v)), 2, voff_range[1]]])
            if len(v) < 3:
                plim[0, 2] = 1.999
                plim[1, 2] = 2.001
            if voff_fixed is not None:
                plim = plim[:, :-1]
            popt, perr = emcee_custom(plim, lnprob, False)
            dof = len(v) - len(popt) - 1
            self.chi2r_mass = -2. * lnprob(popt) / dof
            if voff_fixed is not None:
                popt = np.r_[popt, 0]
                perr = np.r_[perr, 0]
            
            M_p, vb, p_low, voff = popt
            dM_p, dvb, dp_low, dvoff = perr
            Mstar = M_p * unit / sini2
            dMstar = dM_p * unit / sini2
            Mstar /= 0.760  # Appendix A in Aso+15_ApJ_812_27
            dMstar /= 0.760
            p_low = 1 / p_low
            dp_low = p_low**2 * dp_low
            self.popt = popt
            self.perr = perr
            self.Mstar = Mstar
            self.dMstar = dMstar
            print(f'voff = {voff:.3f} +/- {dvoff:.3f}')
            print(f'vb = {vb:.3f} +/- {dvb:.3f}')
            print(f'pout = {p_low:.3f} +/- {dp_low:.3f}')
            print(f'Mstar = {Mstar:.3f} +/- {dMstar:.3f} Msun (1/0.76 corrected)')

    def plot_center(self, pa: float = None,
                     filehead: str = 'channelanalysis',
                     show_figs: bool = False,
                     title: str = None):
        plt.rcParams['font.size'] = 20
        plt.rcParams['axes.linewidth'] = 1.5
        plt.rcParams['xtick.direction'] = 'out'
        plt.rcParams['ytick.direction'] = 'out'
        plt.rcParams['xtick.major.size'] = 12
        plt.rcParams['ytick.major.size'] = 12
        plt.rcParams['xtick.minor.size'] = 8
        plt.rcParams['ytick.minor.size'] = 8
        plt.rcParams['xtick.major.width'] = 1.5
        plt.rcParams['ytick.major.width'] = 1.5
        plt.rcParams['xtick.minor.width'] = 1.5
        plt.rcParams['ytick.minor.width'] = 1.5
        
        kep = ~np.isnan(self.kepler['xc']) * ~np.isnan(self.kepler['yc'])
        if np.any(kep) > 0:
            vmax = np.abs(np.max(self.v[kep]))
        else:
            vmax = 1
        xmax, ymax = np.max(self.x), np.max(self.y)
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        r = np.linspace(-xmax * 1.5, xmax * 1.5, 5)
        if pa is not None:
            p = np.radians(pa)
            a = rot(0, r, -p)
            ax.plot(a[0], a[1], 'k-')
        p = np.radians(self.pa_grad)
        a = rot(0, r, -p)
        ax.plot(a[0] + self.xoff, a[1] + self.yoff, 'g-')
        a = rot(r / 1.5 * 0.5, 0, -p)
        ax.plot(a[0] + self.xoff, a[1] + self.yoff, 'g-')
        if np.any(kep):
            x, y = self.x, self.y
            z = np.sum(self.data[kep], axis=0) * self.dv
            ax.pcolormesh(x, y, z, shading='nearest', cmap='binary', zorder=1)
        x = self.kepler['xc'] + self.xoff
        y = self.kepler['yc'] + self.yoff
        dx, dy = self.kepler['dxc'], self.kepler['dyc']
        ax.errorbar(x, y, xerr=dx, yerr=dy, color='g', fmt='.', zorder=4)
        w = self.v * 1
        if self.popt is not None:
            w = w - self.popt[3]
        m = ax.scatter(x, y, c=w, cmap='jet', s=50,
                       vmin=-vmax, vmax=vmax, zorder=5)
        x = self.center['xc']
        y = self.center['yc']
        dx, dy = self.center['dxc'], self.center['dyc']
        ax.errorbar(x, y, xerr=dx, yerr=dy, color='g', fmt='.', zorder=2)
        ax.scatter(x, y, c=self.v, cmap='jet', s=50,
                   vmin=-vmax, vmax=vmax, zorder=3)
        ax.scatter(x, y, c='w', s=15, zorder=3)
        fig.colorbar(m, ax=ax, label=r'velocity (km s$^{-1}$)')
        bpos = xmax - 0.7 * self.bmaj
        e = Ellipse((bpos, -bpos), width=self.bmin, height=self.bmaj,
                    angle=self.bpa * np.sign(self.dx), facecolor='khaki',
                    zorder=1)
        ax.add_patch(e)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('R.A. offset (au)')
        ax.set_ylabel('Dec. offset (au)')
        ax.set_xlim(xmax * 1.01, -xmax * 1.01)  # au
        ax.set_ylim(-ymax * 1.01, ymax * 1.01)  # au
        ax.set_xticks(ax.get_yticks())
        ax.set_yticks(ax.get_xticks())
        ax.set_xlim(xmax * 1.01, -xmax * 1.01)  # au
        ax.set_ylim(-ymax * 1.01, ymax * 1.01)  # au
        if title is not None:
            ax.set_title(title)
        fig.tight_layout()
        fig.savefig(filehead + '.radec.png', transparent=True)
        if show_figs: plt.show()
        plt.close()
        print(f'- Plotted in {filehead}.radec.png')
        
        x, y = self.kepler['xc'], self.kepler['yc']
        dx, dy = self.kepler['dxc'], self.kepler['dyc']
        x = np.abs(x * np.sin(p) + y * np.cos(p))
        dx = np.hypot(dx * np.sin(p), dy * np.cos(p))
        w = self.v * 1
        if self.popt is not None:
            w = w - self.popt[3]
        v = np.abs(w)
        xn = self.center['xc'] - self.xoff
        yn = self.center['yc'] - self.yoff
        dxn, dyn = self.center['dxc'], self.center['dyc']
        xn = np.abs(xn * np.sin(p) + yn * np.cos(p))
        dxn = np.hypot(dxn * np.sin(p), dyn * np.cos(p))

        if np.any(kep):
            x0 = np.exp(np.mean(np.log(x[kep])))
            v0 = np.exp(np.mean(np.log(v[kep])))
            ratiox = np.exp(np.max(np.abs(np.log(x[kep] / x0))))
            ratiov = np.exp(np.max(np.abs(np.log(v[kep] / v0))))
            ratio = max(ratiox, ratiov, 3.2)
            xmin = x0 / ratio
            xmax = x0 * ratio
            vmin = v0 / ratio
            vmax = v0 * ratio
        else:
            xmax = np.max(self.x) * 1.5
            xmin = xmax / 30
            vmax = np.max(self.v) * 1.5
            vmin = vmax / 30
        
        fig = plt.figure(figsize=(5.3, 5.3))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(xmin * 0.99, xmax * 1.01)  # au
        ax.set_ylim(vmin * 0.99, vmax * 1.01)  # km/s
        ax.errorbar(x, v, xerr=dx, fmt='o', color='k', zorder=2)
        ax.plot(x[w < 0], v[w < 0], 'bo', zorder=3)
        ax.plot(x[w > 0], v[w > 0], 'ro', zorder=3)
        ax.plot(xn[w < 0], v[w < 0], 'bo', zorder=1)
        ax.plot(xn[w < 0], v[w < 0], 'wo', zorder=1, markersize=4)
        ax.plot(xn[w > 0], v[w > 0], 'ro', zorder=1)
        ax.plot(xn[w > 0], v[w > 0], 'wo', zorder=1, markersize=4)
        if ~np.isnan(self.Mstar):
            vp = np.abs(v[~np.isnan(x)])
            if vp.max() - vp.min() < 0.1:
                vp = (vp.max() + vp.min()) / 2.
                vp = np.array([vp - 0.05, vp + 0.05])
            vp = np.geomspace(vp.min(), vp.max(), 100)
            M_p, v_break, p_low, _ = self.popt
            rp = r_kep_out(vp, M_p, v_break, p_low, 0)
            ax.plot(rp, vp, 'g-', zorder=4)
        ax.set_xscale('log')
        ax.set_yscale('log')
        def nice_ticks(ticks, tlim):
            order = 10**np.floor(np.log10(tlow := tlim[0]))
            tlow = np.ceil(tlow / order) * order
            order = 10**np.floor(np.log10(tup := tlim[1]))
            tup = np.floor(tup / order) * order
            return np.sort(np.r_[ticks, tlow, tup])
        xticks = nice_ticks(ax.get_xticks(), (xmin, xmax))
        yticks = nice_ticks(ax.get_yticks(), (vmin, vmax))
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        def nice_labels(ticks):
            digits = np.floor(np.log10(ticks)).astype('int').clip(None, 0)
            return [f'{t:.{d:d}f}' for t, d in zip(ticks, -digits)]
        ax.set_xticklabels(nice_labels(xticks))
        ax.set_yticklabels(nice_labels(yticks))
        ax.set_xlim(xmin * 0.99, xmax * 1.01)  # au
        ax.set_ylim(vmin * 0.99, vmax * 1.01)  # km/s
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Offset along v-grad (au)')
        ax.set_ylabel(r'Velocity (km s$^{-1}$)')
        if title is not None:
            ax.set_title(title)
        fig.tight_layout()
        fig.savefig(filehead + '.majvel.png', transparent=True)
        if show_figs: plt.show()
        plt.close()
        print(f'- Plotted in {filehead}.majvel.png')
