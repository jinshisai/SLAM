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
from scipy.optimize import differential_evolution as diffevo
from scipy.interpolate import interp1d

from utils import emcee_corner
from pvanalysis.analysis_tools import doublepower_r

GG = constants.G.si.value
M_sun = constants.M_sun.si.value
au = units.au.to('m')
unit = 1.e6 * au / GG / M_sun
        
def rot(x, y, pa):
    s = x * np.cos(pa) - y * np.sin(pa)  # along minor axis
    t = x * np.sin(pa) + y * np.cos(pa)  # along major axis
    return np.array([s, t])

def gauss2d(xy, peak, cx, cy, wx, wy, pa):
    x, y = xy
    s, t = rot(x - cx, y - cy, pa)
    return np.ravel(peak * np.exp2(-s**2 / wx**2 - t**2 / wy**2))



class TwoDGrad():

    #def __init__(self):


    def read_cubefits(self, cubefits: str, center: str = None,
                      dist: float = 1, vsys: float = 0,
                      xmin: float = None, xmax: float = None,
                      ymin: float = None, ymax: float = None,
                      vmin: float = None, vmax: float = None,
                      xskip: int = 1, yskip: int = 1,
                      sigma: float = None,
                      centering_velocity: bool = False) -> dict:
        """
        Read a position-velocity diagram in the FITS format.

        Parameters
        ----------
        cubefits : str
            Name of the input FITS file including the extension.
        center : str
            Coordinates of the target: e.g., "01h23m45.6s 01d23m45.6s".
        dist : float
            Distance of the target, used to convert arcsec to au.
        vsys : float
            Systemic velocity of the target.
        xmin, xmax : float
            The R.A. axis is limited to (xmin, xmax) in the unit of au.
        ymin, ymax : float
            The Dec. axis is limited to (ymin, ymax) in the unit of au.
        vmax : float
            The velocity axis of the PV diagram is limited to (vmin, vmax).
        vmin : float
            The velocity axis of the PV diagram is limited to (vmin, vmax).
        xskip : int
            Skip xskip pixels in the x axis.
        yskip : int
            Skip yskip pixels in the y axis.
        sigma : float
            Standard deviation of the FITS data. None means automatic.
        centering_velocity : bool
            One channel has the exact velocity of vsys by interpolation.

        Returns
        ----------
        fitsdata : dict
            x (1D array), v (1D array), data (2D array), header, and sigma.
        """
        cc = constants.c.si.value
        f = fits.open(cubefits)[0]
        d, h = np.squeeze(f.data), f.header
        if center is None:
            cx, cy = 0, 0
        else:
            c0 = SkyCoord('00h00m00s 00d00m00s', frame='icrs')
            c1 = [h['CRVAL1'] * units.degree, h['CRVAL2'] * units.degree]
            c1 = c0.spherical_offsets_by(*c1)
            c3 = c1.spherical_offsets_to(SkyCoord(center, frame='icrs'))
            cx = c3[0].degree
            cy = c3[1].degree
        if sigma is None:
            sigma = np.mean([np.nanstd(d[:2]), np.std(d[-2:])])
            print(f'sigma = {sigma:.3e}')
        x = (np.arange(h['NAXIS1']) - h['CRPIX1'] + 1) * h['CDELT1']
        y = (np.arange(h['NAXIS2']) - h['CRPIX2'] + 1) * h['CDELT2']
        v = (np.arange(h['NAXIS3']) - h['CRPIX3'] + 1) * h['CDELT3']
        crpix = int(h['CRPIX1']) - 1
        startpix = crpix % xskip
        x = x[startpix::xskip]
        h['CRPIX1'] = (crpix - startpix) // xskip + 1
        h['CDELT1'] = h['CDELT1'] * xskip
        d = d[:, :, startpix::xskip]
        crpix = int(h['CRPIX2']) - 1
        startpix = crpix % yskip
        y = y[startpix::yskip]
        h['CRPIX2'] = (crpix - startpix) // yskip + 1
        h['CDELT2'] = h['CDELT2'] * yskip
        d = d[:, startpix::yskip, :]
        v = v + h['CRVAL3']
        x = (x - cx) * 3600. * dist  # au
        y = (y - cy) * 3600. * dist  # au
        v = (1. - v / h['RESTFRQ']) * cc / 1.e3 - vsys  # km/s
        i0 = 0 if xmax is None else np.argmin(np.abs(x - xmax))
        i1 = len(x) if xmin is None else np.argmin(np.abs(x - xmin))
        j0 = 0 if ymin is None else np.argmin(np.abs(y - ymin))
        j1 = len(y) if ymax is None else np.argmin(np.abs(y - ymax))
        x, y = x[i0:i1 + 1], y[j0:j1 + 1]
        if centering_velocity:
            f = interp1d(v, d, kind='cubic', bounds_error=False,
                         fill_value=0, axis=0)
            d = f(v := v - v[np.argmin(np.abs(v))])
        k0 = 0 if vmin is None else np.argmin(np.abs(v - vmin))
        k1 = len(v) if vmax is None else np.argmin(np.abs(v - vmax))
        self.offpix = (i0, j0, k0)
        v = v[k0:k1 + 1]
        d =  d[k0:k1 + 1, j0:j1 + 1, i0:i1 + 1]
        dx, dy, dv = x[1] - x[0], y[1] - y[0], v[1] - v[0]
        if 'BMAJ' in h.keys():
            bmaj = h['BMAJ'] * 3600. * dist  # au
            bmin = h['BMIN'] * 3600. * dist  # au
            bpa = h['BPA']  # deg
        else:
            bmaj, bmin, bpa = dy, -dx, 0
            print('No valid beam in the FITS file.')
        self.x, self.dx, self.nx = x, dx, len(x)
        self.y, self.dy, self.ny = y, dy, len(y)
        self.v, self.dv, self.nv = v, dv, len(v)
        self.data, self.header, self.sigma = d, h, sigma
        self.bmaj, self.bmin, self.bpa = bmaj, bmin, bpa
        self.cubefits, self.dist, self.vsys = cubefits, dist, vsys
        return {'x':x, 'y':y, 'v':v, 'data':d, 'header':h, 'sigma':sigma}


    def get_mom1grad(self, cutoff: float = 5, vmask: list = [0, 0],
                     weights: np.ndarray = 1):
        self.make_moment01(vmask=vmask)
        mom0, mom1, sigma_mom0 = self.mom0, self.mom1, self.sigma_mom0
        mom1[mom0 < cutoff * sigma_mom0] = np.nan
        x, y = np.meshgrid(self.x, self.y)
        x = x[~np.isnan(mom1)]
        y = y[~np.isnan(mom1)]
        mom0 = mom0[~np.isnan(mom1)]
        mom1 = mom1[~np.isnan(mom1)]
        w = weights
        xx = np.sum(x**2 * w)
        yy = np.sum(y**2 * w)
        xy = np.sum(x * y * w)
        x1 = np.sum(x * w)
        y1 = np.sum(y * w)
        one = np.sum(w)
        m = [[xx, xy, x1],
             [xy, yy, y1],
             [x1, y1, one]]
        a = [np.sum(mom1 * x * w), np.sum(mom1 * y * w), np.sum(mom1 * w)]
        a, b, v0 = np.dot(np.linalg.inv(m), a)
        amp_g = np.hypot(a, b)
        ang_g = np.arctan2(a, b)
        roff = -v0 / amp_g
        self.xoff = x0 = roff * np.sin(ang_g)
        self.yoff = y0 = roff * np.cos(ang_g)
        ang_g = np.degrees(ang_g)
        print(f'Gradient amp. : {amp_g:.2e} km/s/au')
        print(f'Gradient angle: {ang_g:.2f} degree')
        print(f'Offset: v={v0:.2f} km/s or (x,y)=({x0:.2f}, {y0:.2f}) au')
        return {'xoff':x0, 'yoff':y0, 'amp':amp_g, 'ang':ang_g}

        
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

        def chi2(x, x_in, y_in, dx_in, dy_in):
            xoff, yoff, pa = x
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
            return np.sum(d1 + d2 + d3)
            
        def low_velocity(x_in, y_in, pa):
            parad = np.radians(pa)
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
                imax = np.nanargmax(np.abs(x * cospa + y * cospa))
                c[n0:n0+1 + imax] = True
            return c.astype('bool')

        goodsolution = False
        xoff = yoff = pa_grad = np.nan
        while not goodsolution:
            if np.all(np.isnan(xc) | np.isnan(yc)):
                print('No point survived.')
                break
            xmin = -1e-6 if fixcenter else self.x.min()
            ymin = -1e-6 if fixcenter else self.y.min()
            xmax = 1e-6 if fixcenter else self.x.max()
            ymax = 1e-6 if fixcenter else self.y.max()
            bounds = [[xmin, xmax], [ymin, ymax], [pa0 - 90.0, pa0 + 90.0]]
            res = diffevo(func=chi2, bounds=bounds,
                          args=[xc, yc, dxc, dyc], x0=[0, 0, pa0])
            xoff, yoff, pa_grad = res.x
            print(f'xoff, yoff, pa = {xoff:.2f} au, {yoff:.2f} au, {pa_grad:.2f} deg')
            if lowvelfilter:                
                c1 = low_velocity(xc - xoff, yc - yoff, pa_grad)
                xc[c1] = yc[c1] = dxc[c1] = dyc[c1] = np.nan

            res = diffevo(func=chi2, bounds=bounds,
                          args=[xc, yc, dxc, dyc], x0=[0, 0, pa0])
            xoff, yoff, pa_grad = res.x
            print(f'xoff, yoff, pa = {xoff:.2f} au, {yoff:.2f} au, {pa_grad:.2f} deg')

            c1 = bad_channels(xc, yc, xoff, yoff, pa_grad)
            if np.any(c1):
                xc[c1] = yc[c1] = dxc[c1] = dyc[c1] = np.nan
            else:
                goodsolution = True
        
        xc, yc = xc - xoff, yc - yoff
        self.xoff, self.yoff = xoff, yoff
        self.pa_grad = pa_grad
        self.kepler = {'xc':xc, 'dxc':dxc, 'yc':yc, 'dyc':dyc}
        

    def calc_mstar(self, incl: float = 90,
                   minabserr: float = 0.1, minrelerr: float = 0.01):
        self.incl = incl
        xc = self.kepler['xc'] * 1
        yc = self.kepler['yc'] * 1
        dxc = self.kepler['dxc'] * 1
        dyc = self.kepler['dyc'] * 1
        self.Rkep = np.nan
        self.Vkep = np.nan
        self.vmid = np.nan
        self.Mstar = np.nan
        self.popt = np.nan
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
                r_break, v_break, dp = p
                r_model = doublepower_r(v=v, r_break=r_break, v_break=v_break,
                                        p_in=0.5, dp=dp, vsys=0)
                chi2 = np.sum(((r - s_model * r_model) / dr)**2)
                return -0.5 * chi2
            plim = np.array([[np.min(np.abs(r)), np.min(np.abs(v)), 0],
                             [np.max(np.abs(r)), np.max(np.abs(v)), 10]])
            popt, perr = emcee_corner(plim, lnprob,
                                      nwalkers_per_ndim=16,
                                      nburnin= 2000, nsteps=2000,
                                      labels=['Vb (km/s)', 'Rb (au)', 'dp'],
                                      rangelevel=0.8, simpleoutput=True)
            rb, vb, dp = popt
            drb, dvb, ddp = perr
            Mstar = rb * vb**2 * unit / np.sin(np.radians(incl))**2
            Mstar /= 0.760  # Appendix A in Aso+15_ApJ_812_27
            dMstar = Mstar * np.sqrt((drb / rb)**2 + (2 * dvb / vb)**2)
            self.popt = popt
            self.Mstar = Mstar
            self.dMstar = dMstar
            print(f'pout = {dp+0.5:.3} +/- {ddp:.3}')
            print(f'Mstar = {Mstar:.3f} +/- {dMstar:.3f} Msun (1/0.76 corrected)')
        

    def make_moment01(self, vmask: list = [0, 0]):
        v = self.v * 1
        v[(vmask[0] < v) * (v < vmask[1])] = np.nan
        self.sigma_mom0 = self.sigma * self.dv * np.sqrt(len(v[~np.isnan(v)]))
        v = np.moveaxis([[v]], 2, 0)
        total = np.nansum(self.data + v * 0, axis=0)
        mom1 = np.nansum(self.data * v, axis=0) / total
        self.mom0 = total * self.dv
        self.mom1 = mom1
        

    def plot_center(self, pa: float = None,
                     filehead: str = 'channelanalysis',
                     show_figs: bool = False):
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
        m = ax.scatter(x, y, c=self.v, cmap='jet', s=50,
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
        
        x, y = self.kepler['xc'], self.kepler['yc']
        dx, dy = self.kepler['dxc'], self.kepler['dyc']
        x = np.abs(x * np.sin(p) + y * np.cos(p))
        dx = np.hypot(dx * np.sin(p), dy * np.cos(p))
        v = np.abs(self.v)
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
            ratio = max(ratiox, ratiov, 4)
            xmin = x0 / ratio
            xmax = x0 * ratio
            vmin = v0 / ratio
            vmax = v0 * ratio
        else:
            xmax = np.max(self.x) * 1.5
            xmin = xmax / 30
            vmax = np.max(self.v) * 1.5
            vmin = vmax / 30
        
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.set_xlim(xmin * 0.999, xmax * 1.001)  # au
        ax.set_ylim(vmin * 0.999, vmax * 1.001)  # km/s
        ax.errorbar(x, v, xerr=dx, fmt='o', color='k', zorder=2)
        w = self.v
        ax.plot(x[w < 0], v[w < 0], 'bo', zorder=3)
        ax.plot(x[w > 0], v[w > 0], 'ro', zorder=3)
        ax.plot(xn[w < 0], v[w < 0], 'bo', zorder=1)
        ax.plot(xn[w < 0], v[w < 0], 'wo', zorder=1, markersize=4)
        ax.plot(xn[w > 0], v[w > 0], 'ro', zorder=1)
        ax.plot(xn[w > 0], v[w > 0], 'wo', zorder=1, markersize=4)
        if ~np.isnan(self.Mstar):
            vp = np.abs(v[~np.isnan(x)])
            vp = np.geomspace(vp.min(), vp.max(), 100)
            r_break, v_break, dp = self.popt
            rp = doublepower_r(vp, r_break, v_break, 0.5, dp, 0)
            ax.plot(rp, vp, 'm-', zorder=4)
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
        ax.set_xlabel('Offset along vel. grad. (au)')
        ax.set_ylabel(r'Velocity (km s$^{-1}$)')
        ax.grid()
        fig.tight_layout()
        fig.savefig(filehead + '.majvel.png', transparent=True)
        if show_figs: plt.show()
        plt.close()
        print(f'- Plotted in {filehead}.majvel.png')
