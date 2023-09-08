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
from astropy.io import fits
from astropy import constants, units
from astropy.coordinates import SkyCoord
from scipy.optimize import curve_fit


def gauss2d(xy, peak, cx, cy, wx, wy, pa):
    x, y = xy
    z = ((y - cy) + 1j * (x - cx)) / np.exp(1j * pa)
    t, s = np.real(z), np.imag(z)
    return np.ravel(peak * np.exp2(-s**2 / wx**2 - t**2 / wy**2))

def rot(x, y, pa):
    s = x * np.cos(pa) - y * np.sin(pa)  # along minor axis
    t = x * np.sin(pa) + y * np.cos(pa)  # along major axis
    return np.array([s, t])

def irot(s, t, pa):
    x =  s * np.cos(pa) + t * np.sin(pa)  # along R.A. axis
    y = -s * np.sin(pa) + t * np.cos(pa)  # along Dec. axis
    return np.array([x, y])


class TwoDGrad():

#    def __init__(self):

    def read_cubefits(self, cubefits, dist, center, vsys=0, xmax=1e4, ymax=1e4,
                      vmax=100, sigma=None):
        """
        Read a position-velocity diagram in the FITS format.

        Parameters
        ----------
        cubefits : str
            Name of the input FITS file including the extension.
        dist : float
            Distance of the target, used to convert arcsec to au.
        center : str
            Coordinates of the target: e.g., "01h23m45.6s 01d23m45.6s".
        vsys : float
            Systemic velocity of the target.
        xmax : float
            The R.A. axis is limited to (-xmax, xmax) in the unit of au.
        ymax : float
            The Dec. axis is limited to (-xmax, xmax) in the unit of au.
        vmax : float
            The velocity axis of the PV diagram is limited to (-vmax, vmax).
        sigma : float
            Standard deviation of the FITS data. None means automatic.

        Returns
        ----------
        fitsdata : dict
            x (1D array), v (1D array), data (2D array), header, and sigma.
        """
        cc = constants.c.si.value
        coord = SkyCoord(center, frame='icrs')
        cx, cy = coord.ra.degree, coord.dec.degree
        f = fits.open(cubefits)[0]
        d, h = np.squeeze(f.data), f.header
        if sigma is None:
            sigma = np.mean([np.nanstd(d[:2]), np.std(d[-2:])])
            print(f'sigma = {sigma:.3e}')
        x = (np.arange(h['NAXIS1'])-h['CRPIX1']+1)*h['CDELT1']+h['CRVAL1']
        y = (np.arange(h['NAXIS2'])-h['CRPIX2']+1)*h['CDELT2']+h['CRVAL2']
        v = (np.arange(h['NAXIS3'])-h['CRPIX3']+1)*h['CDELT3']+h['CRVAL3']
        x = (x - cx) * 3600. * dist  # au
        y = (y - cy) * 3600. * dist  # au
        v = (1. - v / h['RESTFRQ']) * cc / 1.e3 - vsys  # km/s
        i0, i1 = np.argmin(np.abs(x - xmax)), np.argmin(np.abs(x + xmax))
        j0, j1 = np.argmin(np.abs(y + ymax)), np.argmin(np.abs(y - ymax))
        k0, k1 = np.argmin(np.abs(v + vmax)), np.argmin(np.abs(v - vmax))
        x, y, v = x[i0:i1 + 1], y[j0:j1 + 1], v[k0:k1 + 1],
        d =  d[k0:k1 + 1, j0:j1 + 1, i0:i1 + 1]
        dx, dy, dv = x[1] - x[0], y[1] - y[0], v[1] - v[0]
        if 'BMAJ' in h.keys():
            bmaj = h['BMAJ'] * 3600. * dist  # au
            bmin = h['BMIN'] * 3600. * dist  # au
            bpa = h['BPA']  # deg
        else:
            bmaj, bmin, bpa = dy, -dx, 0
            print('No valid beam in the FITS file.')
        self.x, self.dx = x, dx
        self.y, self.dy = y, dy
        self.v, self.dv = v, dv
        self.data, self.header, self.sigma = d, h, sigma
        self.bmaj, self.bmin, self.bpa = bmaj, bmin, bpa
        self.cubefits, self.dist, self.vsys = cubefits, dist, vsys
        return {'x':x, 'y':y, 'v':v, 'data':d, 'header':h, 'sigma':sigma}

    def get_2Dcenter(self, cubefits=None, dist=None, center=None, vsys=None,
                     xmax=1e4, ymax=1e4, vmax=100, vmin=0, sigma=None,
                     cutoff=5, minrelerr=0.01, minabserr=0.1, method='mean'):
        if not (cubefits is None):
            self.read_cubefits(cubefits, dist, center, vsys,
                               xmax, ymax, vmax, sigma)
        x, y, v = self.x, self.y, self.v
        self.rmax = np.max(np.hypot(*np.meshgrid(x, y)))
        dx, dy, dv = self.dx, self.dy, self.dv
        sigma, d = self.sigma, self.data

        def clipped_error(err, val):
            return max(err, minrelerr * np.abs(val), minabserr * self.bmaj)

        X, Y = np.meshgrid(x, y)
        xc, yc, dxc, dyc = [], [], [], []
        for d_ch in d:
            cond = d_ch > cutoff * sigma
            dd, XX, YY = d_ch[cond], X[cond], Y[cond]
            xval, xerr, yval, yerr = np.nan, np.nan, np.nan, np.nan
            if len(dd) > 1:
                if method == 'mean':
                    xval = np.sum(dd * XX) / np.sum(dd)
                    xerr = sigma * np.sum(XX - xval) / np.sum(dd)
                    yval = np.sum(dd * YY) / np.sum(dd)
                    yerr = sigma * np.sum(YY - yval) / np.sum(dd)
                elif method == 'peak':
                    xval = XX[np.argmax(dd)]
                    xerr = self.bmaj / (np.max(dd) / sigma)
                    yval = YY[np.argmax(dd)]
                    yerr = self.bmaj / (np.max(dd) / sigma)
                elif method == 'gauss':
                    if len(dd) < 7: continue
                    bounds = [[0, -xmax, -ymax, dx, dy, 0],
                              [dd.max() * 2, xmax, ymax, xmax, ymax, np.pi]]
                    try:
                        popt, pcov = curve_fit(gauss2d,
                                               (XX.ravel(), YY.ravel()),
                                               dd.ravel(), max_nfev=1000,
                                               sigma=XX.ravel() * 0 + sigma,
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
        cond = ~np.isnan(xc) * ~np.isnan(yc)
        xc, dxc, yc, dyc, v = np.c_[xc, dxc, yc, dyc, v][cond].T
        self.center = {'v':v, 'xc':xc, 'dxc':dxc, 'yc':yc, 'dyc':dyc}

    def find_rkep(self, pa=0., tol_kep=0.5):
        ### Coordinate transformation ###
        p = np.radians(pa)
        tol = tol_kep * self.bmaj
        self.pa = pa
        self.tol_kep = tol_kep
        v, xc, yc = [self.center[i] for i in ['v', 'xc', 'yc']]
        dxc, dyc = [self.center[i] for i in ['dxc', 'dyc']]
        min_off, maj_off = rot(xc, yc, p) 
        #self.min_off = min_off = np.mean(np.r_[min_off[:3], min_off[-3:]])
        self.min_off = min_off = 0
        #self.maj_off = maj_off = np.mean(np.r_[maj_off[:3], maj_off[-3:]])
        self.maj_off = maj_off = 0
        self.xoff, self.yoff = xoff, yoff = irot(min_off, maj_off, p)
        print(f'min_off, maj_off = {min_off:.2f} au, {maj_off:.2f} au')
        print(f'xoff, yoff = {xoff:.2f} au, {yoff:.2f} au')
        sc, tc = rot(xc - xoff, yc - yoff, p)
        dsc = np.sqrt(np.cos(p)**2 * dxc**2 + np.sin(p)**2 * dyc**2)
        dtc = np.sqrt(np.sin(p)**2 * dxc**2 + np.cos(p)**2 * dyc**2)
        self.minmaj = {'v':v, 'min':sc, 'dmin':dsc, 'maj':tc, 'dmaj':dtc}
        # Find the Keplerian disk radius
        s_s = np.sign(np.mean(sc[v > 0]))
        t_s = np.sign(np.mean(tc[v > 0]))
        sc = [s_s * sc[v > 0], -s_s * sc[v < 0][::-1]]
        tc = [t_s * tc[v > 0], -t_s * tc[v < 0][::-1]]
        dsc = [dsc[v > 0], dsc[v < 0][::-1]]
        dtc = [dtc[v > 0], dtc[v < 0][::-1]]
        xc = [xc[v > 0], xc[v < 0][::-1]]
        yc = [yc[v > 0], yc[v < 0][::-1]]
        dxc = [dxc[v > 0], dxc[v < 0][::-1]]
        dyc = [dyc[v > 0], dyc[v < 0][::-1]]
        v = [v[v > 0], -v[v < 0][::-1]]
        self.__X = {'v':v, 'x':xc, 'dx':dxc, 'y':yc, 'dy':dyc}
        self.__M = {'v':v, 'min':sc, 'dmin':dsc, 'maj':tc, 'dmaj':dtc}
        k_back = [np.argmax(tc[i]) - 1 for i in [0, 1]]
        k_shift = [np.where(np.r_[[1e10], np.abs(sc[i])] > tol)[0][-1] for i in [0, 1]]
        k_shift = [max(0, i - 1) for i in k_shift]
        k_notkep = [max(k_back[i], k_shift[i]) for i in [0, 1]]
        self.k_notkep = k_notkep
        for i, c in zip([0, 1], ['red', 'blue']):
            if k_back[i] == len(sc[i]) - 1:
                print('!!!No spin-up point on the ' + c + 'shifted side.!!!')
            if k_shift[i] == len(sc[i]) - 1:
                print('!!!No major-axis point on the ' + c + 'shifted side.!!!')

        Rkep = np.array([tc[i][k_back[i] + 1] for i in [0, 1]])
        Vkep = np.array([v[i][k_back[i] + 1] for i in [0, 1]])
        for i in [0, 1]:
            if k_back[i] < k_shift[i] < len(sc[i]) - 1:
                Rkep[i] = (tc[i][k_shift[i] + 1] - tc[i][k_shift[i]]) \
                          / (sc[i][k_shift[i] + 1] - sc[i][k_shift[i]]) \
                          * (tol - sc[i][k_shift[i]]) + tc[i][k_shift[i]]
                Vkep[i] = v[i][k_shift[i] + 1]
        Rkep /= 0.760  # Appendix A in Aso+15_ApJ_812_27
        print(f'Rkep(red, blue) = {Rkep[0]:.2f}, {Rkep[1]:.2f} au '
              + '(1/0.76 corrected)')
        print(f'Vkep(red, blue) = {Vkep[0]:.3f}, {Vkep[1]:.3f} km/s ')

    def write_2Dcenter(self, filehead='channelanalysis'):
        """
        Write the 2D center in a text file.

        Parameters
        ----------
        filehead : str
            The output text file has a name of "filehead".center.dat. The file consists of five columns of v (km/s), x (au), dx (au), y (au), and dy (au).
        """
        for title, a in zip(['center', 'minmaj'], [self.center, self.minmaj]):
            if hasattr(self, title):
                np.savetxt(filehead + '.' + title + '.dat',
                           np.array([a[i] for i in a.keys()]).T,
                           header='v (km/s), x (au), dx (au), y (au), dy (au)')
            print(f'- Wrote to {filehead}.' + title + '.dat.')

    def make_moment0(self):
        self.mom0 = np.sum(self.data, axis=0) * self.dv
        self.sigma_mom0 = self.sigma * self.dv * np.sqrt(len(self.v))

    def get_mstar(self, incl):
        v, tc, dtc = [self.__M[i] for i in ['v', 'maj', 'dmaj']]
        k = self.k_notkep
        v_kep = [v[i][k[i] + 1:] for i in [0, 1]]
        t_kep = [tc[i][k[i] + 1:] for i in [0, 1]]
        dt_kep = [dtc[i][k[i] + 1:] for i in [0, 1]]
        vv = np.r_[np.abs(v_kep[0]), np.abs(v_kep[1])]
        tt = np.r_[t_kep[0], t_kep[1]]
        dtt = np.r_[dt_kep[0], dt_kep[1]]
        if len(dtt) == 0:
            print('Skip Mstar calculation.')
            return 
        v0 = np.average(np.abs(vv), weights=1. / dtt**2)
        power_r = lambda v, r0, p: r0 / (v / v0)**(1. / p)
        popt, pcov = curve_fit(power_r, vv, tt,
                               sigma=vv * 0 + dtt,
                               absolute_sigma=True,
                               bounds=[[0, 0.01], [self.rmax, 10]])
        r0, p, dr0, dp = *popt, *np.sqrt(np.diag(pcov))
        print(f'p = {p:.3f} +/- {dp:.3f}')
        print(f'r0 = {r0:.2f} +/- {dr0:.2f} au at {v0:.3f} km/s')
        GG = constants.G.si.value
        M_sun = constants.M_sun.si.value
        au = units.au.to('m')
        unit = 1.e6 * au / GG / M_sun / np.sin(np.radians(incl))**2
        #M0 = r0 * v0**2 * unit  # M_sun
        M0 = np.mean(np.abs(tt) * vv**2) * unit  # M_sun
        M0 /= 0.760  # Appendix A in Aso+15_ApJ_812_27
        dM0 = np.std(np.abs(tt) * vv**2) * unit  # M_sun
        dM0 /= 0.760  # Appendix A in Aso+15_ApJ_812_27
        #dM0 = dr0 / r0 * M0
        print(f'Mstar = {M0:.3f} +/- {dM0:.3f} Msun (1/0.76 corrected)')

    def plot_center(self, filehead='channelanalysis', show_figs=False,
                    xmax=1e4, ymax=1e4, vmax=10, vmin=0.1):
        p = np.radians(self.pa)
        tol = self.tol_kep * self.bmaj
        v, xc, yc = [self.__X[i] for i in ['v', 'x', 'y']]
        dxc, dyc = [self.__X[i] for i in ['dx', 'dy']]
        sc, tc = [self.__M[i] for i in ['min', 'maj']]
        dsc, dtc = [self.__M[i] for i in ['dmin', 'dmaj']]
        k = np.array(self.k_notkep) + 1
        x_k = np.r_[xc[0][k[0]:], xc[1][k[1]:]]
        dx_k = np.r_[dxc[0][k[0]:], dxc[1][k[1]:]]
        y_k = np.r_[yc[0][k[0]:], yc[1][k[1]:]]
        dy_k = np.r_[dyc[0][k[0]:], dyc[1][k[1]:]]
        s_k = np.r_[sc[0][k[0]:], sc[1][k[1]:]]
        ds_k = np.r_[dsc[0][k[0]:], dsc[1][k[1]:]]
        t_k = np.r_[tc[0][k[0]:], tc[1][k[1]:]]
        dt_k = np.r_[dtc[0][k[0]:], dtc[1][k[1]:]]
        v_k = np.r_[v[0][k[0]:], -v[1][k[1]:]]
        x_n = np.r_[xc[0][:k[0]], xc[1][:k[1]]]
        dx_n = np.r_[dxc[0][:k[0]], dxc[1][:k[1]]]
        y_n = np.r_[yc[0][:k[0]], yc[1][:k[1]]]
        dy_n = np.r_[dyc[0][:k[0]], dyc[1][:k[1]]]
        s_n = np.r_[sc[0][:k[0]], sc[1][:k[1]]]
        ds_n = np.r_[dsc[0][:k[0]], dsc[1][:k[1]]]
        t_n = np.r_[tc[0][:k[0]], tc[1][:k[1]]]
        dt_n = np.r_[dtc[0][:k[0]], dtc[1][:k[1]]]
        v_n = np.r_[v[0][:k[0]], -v[1][:k[1]]]
         
        ### Plot the derived points on three planes. ###
        plt.rcParams['font.size'] = 24
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
        # On the moment 0 image
        self.make_moment0()
        x, y, mom0 = self.x, self.y, self.mom0
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        r = np.linspace(-xmax * 1.5, xmax * 1.5, 100)
        r = np.concatenate((irot(0, r, p), [[np.nan],[np.nan]],
                            irot(r, 0, p)), axis=1)
        ax.plot(r[0] + self.xoff, r[1] + self.yoff, 'k-')
        for s in [1, -1]:
            d = irot(self.min_off + s * tol, self.maj_off, p)
            ax.plot(r[0] + d[0], r[1] + d[1], 'k--')
        ax.pcolor(x, y, mom0, shading='nearest', cmap='binary')
        ax.errorbar(x_k, y_k, xerr=dx_k, yerr=dy_k,
                    fmt='.', color='gray', zorder=1)
        ax.errorbar(x_n, y_n, xerr=dx_n, yerr=dy_n,
                    fmt='.', color='gray', zorder=1)
        ax.scatter(x_n, y_n, s=50, c=v_n, marker='^', cmap='jet',
                   vmin=-vmax, vmax=vmax, zorder=2)
        ax.scatter(x_n, y_n, s=50, c='none', marker='^',
                   edgecolors='k', zorder=3)
        sca = ax.scatter(x_k, y_k, s=50, c=v_k, marker='o', cmap='jet',
                         vmin=-vmax, vmax=vmax, zorder=4)
        cb = plt.colorbar(sca, ax=ax, label=r'velocity (km s$^{-1}$)')
        bpos = xmax - 0.7 * self.bmaj
        e = Ellipse((bpos, -bpos), width=self.bmin, height=self.bmaj,
                    angle=self.bpa * np.sign(self.dx), facecolor='g')
        ax.add_patch(e)
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('R.A. offset (au)')
        ax.set_ylabel('Dec. offset (au)')
        ax.set_xlim(xmax * 1.001, -xmax * 1.001)  # au
        ax.set_ylim(-ymax * 1.001, ymax * 1.001)  # au
        ax.set_xticks(ax.get_yticks())
        ax.set_yticks(ax.get_xticks())
        ax.set_xlim(xmax * 1.001, -xmax * 1.001)  # au
        ax.set_ylim(-ymax * 1.001, ymax * 1.001)  # au
        ax.grid()
        fig.tight_layout()
        fig.savefig(filehead + '.radec.png', transparent=True)
        if show_figs: plt.show()
        plt.close()
        print(f'- Plotted in {filehead}.radec.png')

        # On the major-minor plane
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        for t in [tol, -tol]: ax.axhline(t, color='k', lw=1, ls='--')
        ax.errorbar(t_k, s_k, xerr=ds_k, yerr=dt_k,
                    fmt='.', color='gray', zorder=1)
        ax.errorbar(t_n, s_n, xerr=ds_n, yerr=dt_n,
                    fmt='.', color='gray', zorder=1)
        ax.scatter(t_n, s_n, s=50, c=v_n, marker='^', cmap='jet',
                   vmin=-vmax, vmax=vmax, zorder=2)
        ax.scatter(t_n, s_n, s=50, c='none', marker='^',
                   edgecolors='k', zorder=3)
        sca = ax.scatter(t_k, s_k, s=50, c=v_k, marker='o', cmap='jet',
                         vmin=-vmax, vmax=vmax, zorder=4)
        cb = plt.colorbar(sca, ax=ax, label=r'velocity km s$^{-1}$')
        ax.set_aspect(1, adjustable='box')
        ax.set_xlabel('Major offset (au)')
        ax.set_ylabel('Minor offset (au)')
        #ax.set_xlim(-tol * 3, xmax * 0.501)  # au
        #ax.set_ylim(-tol * 3, ymax * 0.501)  # au
        ax.grid()
        fig.tight_layout()
        fig.savefig(filehead + '.majmin.png', transparent=True)
        if show_figs: plt.show()
        plt.close()
        print(f'- Plotted in {filehead}.majmin.png')

        # On the major-velocity plane
        xmax = xmax / 2
        xmin = xmax * (vmin / vmax)
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        ax.errorbar(t_k, np.abs(v_k), xerr=dt_k, fmt='.',
                    color='gray', zorder=1)
        ax.errorbar(t_n, np.abs(v_n), xerr=dt_n, fmt='.',
                    color='gray', zorder=1)
        ax.scatter(t_n, np.abs(v_n), s=100, c=s_n / self.bmaj,
                   marker='^', cmap='Greens',
                   vmin=0, vmax=2 * self.tol_kep, zorder=2)
        sca = ax.scatter(t_k, np.abs(v_k), s=100, c=s_k / self.bmaj,
                         marker='o', cmap='Greens',
                         vmin=0, vmax=2 * self.tol_kep, zorder=3)
        fig.colorbar(sca, ax=ax, label='Minor offset (beam)')
        ax.plot(t_n[v_n < 0], np.abs(v_n[v_n < 0]), '^',
                mfc='none', mec='b', markersize=10, zorder=4)
        ax.plot(t_n[v_n > 0], np.abs(v_n[v_n > 0]), '^',
                mfc='none', mec='r', markersize=10, zorder=4)
        ax.plot(t_k[v_k < 0], np.abs(v_k[v_k < 0]), 'o', mfc='none',
                mec='b', markersize=10, zorder=4)
        ax.plot(t_k[v_k > 0], np.abs(v_k[v_k > 0]), 'o', mfc='none',
                mec='r', markersize=10, zorder=4)
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
        ax.set_xlim(xmin * 0.999, xmax * 1.001)  # au
        ax.set_ylim(vmin * 0.999, vmax * 1.001)  # km/s
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Major offset (au)')
        ax.set_ylabel(r'Velocity (km s$^{-1}$)')
        ax.grid()
        fig.tight_layout()
        fig.savefig(filehead + '.majvel.png', transparent=True)
        if show_figs: plt.show()
        plt.close()
        print(f'- Plotted in {filehead}.majvel.png')
