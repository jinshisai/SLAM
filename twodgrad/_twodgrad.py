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
from scipy.interpolate import interp1d


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
                      xmax: float = 1e4, ymax: float = 1e4,
                      vmin: float = -100, vmax: float = 100,
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
        xmax : float
            The R.A. axis is limited to (-xmax, xmax) in the unit of au.
        ymax : float
            The Dec. axis is limited to (-xmax, xmax) in the unit of au.
        vmax : float
            The velocity axis of the PV diagram is limited to (vmin, vmax).
        vmin : float
            The velocity axis of the PV diagram is limited to (vmin, vmax).
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
            coord = SkyCoord(center, frame='icrs')
            cx = coord.ra.degree - h['CRVAL1']
            cy = coord.dec.degree - h['CRVAL2']
        if sigma is None:
            sigma = np.mean([np.nanstd(d[:2]), np.std(d[-2:])])
            print(f'sigma = {sigma:.3e}')
        x = (np.arange(h['NAXIS1']) - h['CRPIX1'] + 1) * h['CDELT1']
        y = (np.arange(h['NAXIS2']) - h['CRPIX2'] + 1) * h['CDELT2']
        v = (np.arange(h['NAXIS3']) - h['CRPIX3'] + 1) * h['CDELT3']
        v = v + h['CRVAL3']
        x = (x - cx) * 3600. * dist  # au
        y = (y - cy) * 3600. * dist  # au
        v = (1. - v / h['RESTFRQ']) * cc / 1.e3 - vsys  # km/s
        i0, i1 = np.argmin(np.abs(x - xmax)), np.argmin(np.abs(x + xmax))
        j0, j1 = np.argmin(np.abs(y + ymax)), np.argmin(np.abs(y - ymax))
        x, y = x[i0:i1 + 1], y[j0:j1 + 1]
        if centering_velocity:
            f = interp1d(v, d, kind='cubic', bounds_error=False,
                         fill_value=0, axis=0)
            d = f(v := v - v[np.argmin(np.abs(v))])
        k0, k1 = np.argmin(np.abs(v - vmin)), np.argmin(np.abs(v - vmax))
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
        
        
    def filtering(self):
        xc = self.center['xc'] * 1
        yc = self.center['yc'] * 1
        dxc = self.center['dxc'] * 1
        dyc = self.center['dyc'] * 1
        self.xoff = np.nan
        self.yoff = np.nan
        self.pa_grad = np.nan
        if (n := len(self.v)) % 2 == 0:
            print('!!! Even number channels.!!!')
        for i in range(n):
            j = -1 - i
            if np.isnan(xc[i]) or np.isnan(yc)[i]:
                xc[i] = yc[i] = dxc[i] = dyc[i] = np.nan
                xc[j] = yc[j] = dxc[j] = dyc[j] = np.nan
        if not np.any(~np.isnan(xc) * ~np.isnan(yc)):
                print('No blue-red pair.')
                
        goodcenter = False
        xofforg, yofforg = 0, 0
        while not goodcenter:
            if not np.any(c := ~np.isnan(xc) * ~np.isnan(yc)):
                print('Failed to find a good center.')
                break
            self.xoff = xoff = np.median(xc[c])
            self.yoff = yoff = np.median(yc[c])
            print(f'(xoff, yoff) = ({xoff:.2f}, {yoff:.2f}) au')
            x, y = xc - xoff, yc - yoff
            x, y = x + x[::-1], y + y[::-1]
            sx, sy = np.nanstd(x), np.nanstd(y)
            c1 = np.hypot(x / sx, y / sy) > 3.41  # 3.41 covers 99.7%
            c2 = np.hypot(xoff - xofforg, yoff - yofforg) > 1.0  # 1.0 au
            if c2 and np.any(c1):
                xc[c1] = yc[c1] = dxc[c1] = dyc[c1] = np.nan
            else:
                goodcenter = True
                xc = xc - xoff
                yc = yc - yoff
            xofforg, yofforg = xoff, yoff
        
        if np.any(~np.isnan(xc) * ~np.isnan(yc)):
            nhalf = (n - 1) // 2
            b = np.abs(xc[:nhalf])
            wb = (b / dxc[:nhalf])**2
            lnb = np.log(b)
            r = np.abs(xc[-1:-1-nhalf:-1])
            wr = (r / dxc[-1:-1-nhalf:-1])**2
            lnr = np.log(r)
            x1 = np.exp((lnb * wb + lnr * wr) / (wb + wr))
            b = np.abs(yc[:nhalf])
            wb = (b / dyc[:nhalf])**2
            lnb = np.log(b)
            r = np.abs(yc[-1:-1-nhalf:-1])
            wr = (r / dyc[-1:-1-nhalf:-1])**2
            lnr = np.log(r)
            y1 = np.exp((lnb * wb + lnr * wr) / (wb + wr))
            imax = np.nanargmax(np.hypot(x1, y1)) + 1
            jmax = -imax
            xc[imax:jmax] = np.nan
            yc[imax:jmax] = np.nan
            
        goodangle = False
        gradangleorg = 0
        while not goodangle:
            if not np.any(c := ~np.isnan(xc) * ~np.isnan(yc)):
                print('No point seems aligned.')
                gradangle = np.nan
                break
            x, y, dx, dy = xc[c], yc[c], dxc[c], dyc[c]
            xx = np.sum(x * x / (dx * dx))
            yy = np.sum(y * y / (dy * dy))
            xy = np.sum(x * y / (dx * dy))
            gradangle = 0.5 * np.arctan2(2 * xy, yy - xx)
            self.pa_grad = np.degrees(gradangle)
            print(f'Vel. grad.: P.A. = {self.pa_grad:.2f} deg')
            d = xc * np.cos(gradangle) - yc * np.sin(gradangle)
            s = np.nanstd(d)
            c1 = np.abs(d / s) > 3.0  # 3.0 covers 99.7%
            c2 = np.abs(gradangle - gradangleorg) > np.radians(1.0)  # 1.0 deg
            if c2 and np.any(c1):
                xc[c1] = yc[c1] = dxc[c1] = dyc[c1] = np.nan
            else:
                goodangle = True
        
        if np.any(~np.isnan(xc) * ~np.isnan(yc)):
            xmajor = xc * np.sin(gradangle) + yc * np.cos(gradangle)
            dxmajor = np.hypot(dxc * np.sin(gradangle), dyc * np.cos(gradangle))
            nhalf = (n - 1) // 2
            b = np.abs(xmajor[:nhalf])
            wb = (b / dxmajor[:nhalf])**2
            lnb = np.log(b)
            r = np.abs(xmajor[-1:-1-nhalf:-1])
            wr = (r / dxmajor[-1:-1-nhalf:-1])**2
            lnr = np.log(r)
            xmajor = np.exp((lnb * wb + lnr * wr) / (wb + wr))
            imax = np.nanargmax(xmajor) + 1
            jmax = -imax
            xc[imax:jmax] = np.nan
            yc[imax:jmax] = np.nan
        self.kepler = {'xc':xc, 'dxc':dxc, 'yc':yc, 'dyc':dyc}
        

    def calc_mstar(self, incl: float = 90):
        self.incl = incl
        xc = self.kepler['xc'] * 1
        yc = self.kepler['yc'] * 1
        dxc = self.kepler['dxc'] * 1
        dyc = self.kepler['dyc'] * 1
        self.Rkep = np.nan
        self.Vkep = np.nan
        self.vmid = np.nan
        self.Mstar = np.nan
        self.power = np.nan
        if not np.any(c := ~np.isnan(xc) * ~np.isnan(yc)):
            print('No point to calculate Rkep, Vkep, and Mstar.')
        else:
            v, x, y, dx, dy = self.v[c], xc[c], yc[c], dxc[c], dyc[c]
            sin_g = np.sin(np.radians(self.pa_grad))
            cos_g = np.cos(np.radians(self.pa_grad))
            r = np.abs(x * sin_g + y * cos_g)
            v = np.abs(v)
            Rkep = np.max(r) / 0.760  # Appendix A in Aso+15_ApJ_812_27
            Vkep = np.min(v)
            print(f'Max r = {Rkep:.1f} au at v={Vkep:.2f} km/s'
                  + ' (1/0.76 corrected)')
            self.Rkep = Rkep
            self.Vkep = Vkep
            lnr = np.log(r)
            dr = np.hypot(dx * sin_g, dy * cos_g)
            dlnr = dr / r
            lnv0 = np.mean(np.log(v))
            v0 = np.exp(lnv0)
            self.vmid = v0
            lnv = np.log(v) - lnv0
            weights = 1 / dlnr**2
            a00 = np.sum(lnv**2 * weights)
            a01 = a10 = np.sum(lnv * weights)
            a11 = np.sum(weights)
            b0 = np.sum(lnr * lnv * weights)
            b1 = np.sum(lnr * weights)
            #if a00 * a11 - a01 * a10 != 0:
            #    ainv = np.linalg.inv([[a00, a01], [a10, a11]])
            #    a = np.dot([b0, b1], ainv)
            #    da = np.sqrt(np.diag(ainv))
            #else:
            #    a = [np.nan, np.nan]
            #    da = [np.nan, np.nan]
            #    print('Power-law fitting failed.')
            #p = 1 / a[0]
            #dp = da[0] / a[0]**2
            #Mstar = np.exp(a[1] + 2 * lnv0) * unit / np.sin(np.radians(incl))**2
            #Mstar /= 0.760  # Appendix A in Aso+15_ApJ_812_27
            #dMstar = Mstar * 2 * lnv0 * da[1]
            Mstar = np.exp((b1 + 2 * a01) / a11) \
                    * unit / np.sin(np.radians(incl))**2
            Mstar /= 0.760  # Appendix A in Aso+15_ApJ_812_27
            dMstar = np.sqrt(1 / a11) * Mstar
            p, dp = -0.5, 0.0
            self.power = p
            self.Mstar = Mstar
            print(f'Power law index: p = {p:.3} +/- {dp:.3}')
            print(f'Mstar = {Mstar:.3f} +/- {dMstar:.3f} Msun at v={v0:.2f} km/s'
                  + ' (1/0.76 corrected)')
        

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
            ax.pcolormesh(x, y, z, cmap='binary', zorder=1)
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
        ax.plot(self.xoff, self.yoff, 'g+', markersize=10)
        fig.colorbar(m, ax=ax, label=r'velocity (km s$^{-1}$)')
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
            vp = v[w > 0][~np.isnan(x[w > 0])]
            rp = self.Mstar / unit * np.sin(np.radians(self.incl))**2 * 0.760 \
                 / self.vmid**2 * (vp / self.vmid)**(1 / self.power)
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
