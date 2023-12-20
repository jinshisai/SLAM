# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Yusuke Aso
# Created Date: 2022 Jan 27
# version = alpha
# ---------------------------------------------------------------------------
"""
This script makes position-velocity diagrams along the major anc minor axes and reproduces their silhouette by the UCM envelope.
The main class PVSilhouette can be imported to do each steps separately.

Note. FITS files with multiple beams are not supported. The dynamic range for xlim_plot and vlim_plot should be >10 for nice tick labels.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import constants, units
from astropy.coordinates import SkyCoord
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator as RGI
from tqdm import tqdm
import warnings

from utils import emcee_corner
from pvsilhouette.ulrichenvelope import velmax

warnings.simplefilter('ignore', RuntimeWarning)


class PVSilhouette():

#    def __init__(self):

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

    def read_pvfits(self, pvfits: str,
                    dist: float = 1, vsys: float = 0,
                    xmax: float = 1e4,
                    vmin: float = -100, vmax: float = 100,
                    sigma: float = None) -> dict:
        """
        Read a position-velocity diagram in the FITS format.

        Parameters
        ----------
        pvfits : str
            Name of the input FITS file including the extension.
        dist : float
            Distance of the target, used to convert arcsec to au.
        vsys : float
            Systemic velocity of the target.
        xmax : float
            The positional axis is limited to (-xmax, xmax) in the unit of au.
        vmin : float
            The velocity axis is limited to (-vmax, vmax) in the unit of km/s.
        vmax : float
            The velocity axis is limited to (-vmax, vmax) in the unit of km/s.
        sigma : float
            Standard deviation of the FITS data. None means automatic.

        Returns
        ----------
        fitsdata : dict
            x (1D array), v (1D array), data (2D array), header, and sigma.
        """
        cc = constants.c.si.value
        f = fits.open(pvfits)[0]
        d, h = np.squeeze(f.data), f.header
        if sigma is None:
            sigma = np.mean([np.std(d[:2, 10:-10]), np.std(d[-2:, 10:-10]),
                             np.std(d[2:-2, :10]), np.std(d[2:-2, -10:])])
            print(f'sigma = {sigma:.3e}')
        x = (np.arange(h['NAXIS1'])-h['CRPIX1']+1)*h['CDELT1']+h['CRVAL1']
        v = (np.arange(h['NAXIS2'])-h['CRPIX2']+1)*h['CDELT2']+h['CRVAL2']
        x = x * dist  # au
        v = (1. - v / h['RESTFRQ']) * cc / 1.e3 - vsys  # km/s
        i0, i1 = np.argmin(np.abs(x + xmax)), np.argmin(np.abs(x - xmax))
        j0, j1 = np.argmin(np.abs(v - vmin)), np.argmin(np.abs(v - vmax))
        x, v, d = x[i0:i1 + 1], v[j0:j1 + 1], d[j0:j1 + 1, i0:i1 + 1]
        dx, dv = x[1] - x[0], v[1] - v[0]
        if 'BMAJ' in h.keys():
            dNyquist = (bmaj := h['BMAJ'] * 3600. * dist) / 2.  # au
        else:
            dNyquist = bmaj = np.abs(dx)  # au
            print('No valid beam in the FITS file.')
        self.x, self.dx = x, dx
        self.v, self.dv = v, dv
        self.data, self.header, self.sigma = d, h, sigma
        self.bmaj, self.dNyquist = bmaj, dNyquist
        self.pvfits, self.dist, self.vsys = pvfits, dist, vsys
        return {'x':x, 'v':v, 'data':d, 'header':h, 'sigma':sigma}

    def get_PV(self, cubefits: str = None,
               pa: float = 0, center: str = None,
               dist: float = 1, vsys: float = 0,
               xmax: float = 1e4,
               vmin: float = -100, vmax: float = 100,
               sigma: float = None):
        if not (cubefits is None):
            self.read_cubefits(cubefits, center, dist, vsys,
                               -xmax, xmax, -xmax, xmax, vmin, vmax,
                               1, 1, sigma)
        x, y, v = self.x, self.y, self.v
        sigma, d = self.sigma, self.data
        n = np.floor(xmax / self.dy)
        r = (np.arange(2 * n + 1) - n) * self.dy
        ry = r * np.cos(np.radians(pa))
        rx = r * np.sin(np.radians(pa))
        dpvmajor = [None] * len(v)
        dpvminor = [None] * len(v)
        for i in range(len(v)):
            interp = RGI((-x, y), d[i], bounds_error=False)
            dpvmajor[i] = interp((-rx, ry))
            dpvminor[i] = interp((ry, rx))
        self.dpvmajor = np.array(dpvmajor)
        self.dpvminor = np.array(dpvminor)
        self.x = r
    
    def put_PV(self, pvmajorfits: str, pvminorfits: str,
               dist: float, vsys: float,
               rmax: float, vmin: float, vmax: float, sigma: float):
        self.read_pvfits(pvmajorfits, dist, vsys, rmax, vmin, vmax, sigma)
        self.dpvmajor = self.data
        self.read_pvfits(pvminorfits, dist, vsys, rmax, vmin, vmax, sigma)
        self.dpvminor = self.data

    def fitting(self, incl: float = 90,
                Mstar_range: list = [0.01, 10],
                Rc_range: list = [1, 1000],
                alphainfall_range: list = [0.01, 1],
                Mstar_fixed: float = None,
                Rc_fixed: float = None,
                alphainfall_fixed: float = None,
                cutoff: float = 5, vmask: list = [0, 0],
                figname: str = 'PVsilhouette',
                show: bool = False,
                progressbar: bool = True,
                kwargs_emcee_corner: dict = {}):
        Nyquistskip = int(np.round(self.bmaj / self.dx / 2))
        x = self.x[::Nyquistskip]
        vintp = np.linspace(self.v[0], self.v[-1], (len(self.v) - 1) * 10 + 1)
        dvintp = vintp[1] - vintp[0]
        majintp = interp1d(self.v, self.dpvmajor, kind='cubic', axis=0)(vintp)
        minintp = interp1d(self.v, self.dpvminor, kind='cubic', axis=0)(vintp)
        vobs = []
        vobserr = []
        for d in [majintp.T[::Nyquistskip, :], minintp.T[::Nyquistskip, :]]:
            vtmp = []
            dvtmp = []
            for c in d:
                grad = (np.roll(c, -1) - np.roll(c, 1)) / (2 * dvintp)
                cond = (c > cutoff * self.sigma)
                cond = cond * ((vintp < vmask[0]) + (vmask[1] < vintp))
                grad, vgood = grad[cond], vintp[cond]
                err = self.sigma / np.abs(grad)
                if len(vgood) == 0:
                    vtmp.append([np.nan, np.nan])
                    dvtmp.append([np.nan, np.nan])
                else:
                    vtmp.append([np.clip(vgood[0], None, 0), np.clip(vgood[-1], 0, None)])
                    dvtmp.append([err[0], err[-1]])
            vobs.append(vtmp)
            vobserr.append(dvtmp)
        vobs = np.moveaxis(vobs, 1, 2)
        vobserr = np.moveaxis(vobserr, 1, 2)
        def getquad(m):
            nv, nx = np.shape(m)
            q =   np.sum(m[:nv//2, :nx//2]) + np.sum(m[nv//2:, nx//2:]) \
                - np.sum(m[nv//2:, :nx//2]) - np.sum(m[:nv//2, nx//2:])
            return int(np.sign(q))
        majquad = getquad(self.dpvmajor)
        minquad = getquad(self.dpvminor) * (-1)
        def makemodel(Mstar, Rc, alphainfall):
            a = velmax(x, Mstar=Mstar, Rc=Rc,
                       alphainfall=alphainfall, incl=incl)
            vmod = [[a[i][j][::k] for j in ['vlosmin', 'vlosmax']]
                    for i, k in zip(['major', 'minor'], [majquad, minquad])]
            return np.array(vmod)

        p_fixed = np.array([Mstar_fixed, Rc_fixed, alphainfall_fixed])
        if None in p_fixed:
            labels = np.array(['log Mstar', 'log Rc', r'log $\alpha$'])
            labels = labels[p_fixed == None]
            kwargs0 = {'nwalkers_per_ndim':16, 'nburnin':100, 'nsteps':500,
                       'rangelevel':None, 'labels':labels,
                       'figname':figname+'.corner.png', 'show_corner':show}
            kwargs = dict(kwargs0, **kwargs_emcee_corner)
            if progressbar:
                total = kwargs['nwalkers_per_ndim'] * len(p_fixed[p_fixed == None])
                total *= kwargs['nburnin'] + kwargs['nsteps'] + 2
                bar = tqdm(total=total)
                bar.set_description('Within the ranges')
            def lnprob(p):
                if progressbar:
                    bar.update(1)
                q = p_fixed.copy()
                q[p_fixed == None] = 10**p
                chi2 = np.nansum(((vobs - makemodel(*q)) / vobserr)**2)
                return -0.5 * chi2
            plim = np.array([Mstar_range, Rc_range, alphainfall_range])
            plim = np.log10(plim[p_fixed == None]).T
            mcmc = emcee_corner(plim, lnprob, simpleoutput=False, **kwargs)
            popt = p_fixed.copy()
            popt[p_fixed == None] = 10**mcmc[0]
            plow = p_fixed.copy()
            plow[p_fixed == None] = 10**mcmc[1]
            phigh = p_fixed.copy()
            phigh[p_fixed == None] = 10**mcmc[3]
        else:
            popt = p_fixed
            plow = p_fixed
            phigh = p_fixed
        self.popt = popt
        self.plow = plow
        self.phigh = phigh
        print(f'M* = {plow[0]:.2f}, {popt[0]:.2f}, {phigh[0]:.2f} Msun')
        print(f'Rc = {plow[1]:.0f}, {popt[1]:.0f}, {phigh[1]:.0f} au')
        print(f'alpha = {plow[2]:.2f}, {popt[2]:.2f}, {phigh[2]:.2f}')

        a = velmax(x, Mstar=popt[0], Rc=popt[1],
                   alphainfall=popt[2], incl=incl)
        fig = plt.figure()
        ax = fig.add_subplot(1, 2, 1)
        ax.contour(self.x, self.v, self.dpvmajor,
                   levels=np.arange(1, 10) * 3 * self.sigma, colors='k')
        ax.plot(x * majquad, a['major']['vlosmax'], '-r')
        ax.plot(x * majquad, a['major']['vlosmin'], '-r')
        ax.errorbar(x, vobs[0][0], yerr=vobserr[0][0], fmt='ob', ms=2)
        ax.errorbar(x, vobs[0][1], yerr=vobserr[0][1], fmt='ob', ms=2)
        ax.set_xlabel('major offset (au)')
        ax.set_ylabel(r'$V-V_{\rm sys}$ (km s$^{-1}$)')
        ax.set_ylim(np.min(self.v), np.max(self.v))
        ax = fig.add_subplot(1, 2, 2)
        ax.contour(self.x, self.v, self.dpvminor,
                   levels=np.arange(1, 10) * 3 * self.sigma, colors='k')
        ax.plot(x * minquad, a['minor']['vlosmax'], '-r')
        ax.plot(x * minquad, a['minor']['vlosmin'], '-r')
        ax.errorbar(x, vobs[1][0], yerr=vobserr[1][0], fmt='ob', ms=2)
        ax.errorbar(x, vobs[1][1], yerr=vobserr[1][1], fmt='ob', ms=2)
        ax.set_xlabel('minor offset (au)')
        ax.set_ylim(self.v.min(), self.v.max())
        ax.set_title(r'$M_{*}$'+f'={popt[0]:.2f}'+r'$M_{\odot}$'
            +', '+r'$R_{c}$'+f'={popt[1]:.0f} au'
            +'\n'+r'$\alpha$'+f'={popt[2]:.2f}'
            +', '+r'$\alpha ^{2} M_{*}$'+f'={popt[0] * popt[2]**2:.2f}')
        fig.tight_layout()
        fig.savefig(figname + '.model.png')
        if show: plt.show()