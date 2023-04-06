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

############ INPUTS ############
cubefits = './pvsilhouette/IRAS16253_SBLB_C18O_robust_2.0.imsub.fits'
pvmajorfits = './pvsilhouette/IRAS16253_SBLB_C18O_robust_2.0.pvmajor.fits'
pvminorfits = './pvsilhouette/IRAS16253_SBLB_C18O_robust_2.0.pvminor.fits'
center = '16h28m21.61526785s -24d36m24.32538414s'
pa = 113 - 180  # deg
incl = 65  # deg
vsys = 4  # km/s
dist = 139  # pc
sigma = 1.9e-3  # Jy/beam; None means automatic calculation.
cutoff = 5.0  # sigma
rmax = 1 * dist  # au
vlim = (-2.5, 2.5)
vmask = (-0.1, 0.2)
xmax_plot = rmax  # au
ymax_plot = xmax_plot  # au
show_figs = True
minrelerr = 0.01
minabserr = 0.1
method = 'gauss'  # mean or gauss
write_point = False  # True: write the 2D centers to a text file.
################################


import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from astropy.io import fits
from astropy import constants, units
from astropy.coordinates import SkyCoord
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.optimize import curve_fit
import warnings

from utils import emcee_corner
from UlrichEnvelope import velmax

warnings.simplefilter('ignore', RuntimeWarning)

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


class PVSilhouette():

#    def __init__(self):

    def read_cubefits(self, cubefits, dist, center, vsys=0, xmax=1e4, ymax=1e4,
                      vmin=-100, vmax=100, sigma=None):
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
            The velocity axis of the PV diagram is limited to (vmin, vmax).
        vmin : float
            The velocity axis of the PV diagram is limited to (vmin, vmax).
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
        k0, k1 = np.argmin(np.abs(v - vmin)), np.argmin(np.abs(v - vmax))
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

    def read_pvfits(self, pvfits, dist, vsys=0, xmax=1e4,
                    vmin=-100, vmax=100, sigma=None):
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

    def get_PV(self, cubefits=None, pa=0, dist=None, center=None, vsys=None,
               rmax=1e4, vmin=-100, vmax=100, sigma=None, show=False):
        if not (cubefits is None):
            self.read_cubefits(cubefits, dist, center, vsys,
                               rmax, rmax, vmin, vmax, sigma)
        x, y, v = self.x, self.y, self.v
        sigma, d = self.sigma, self.data
        n = np.floor(rmax / self.dy)
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
        self.r = r
        if show:
            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1)
            ax.contour(self.r, self.v, self.dpvmajor,
                       levels=np.arange(1, 10) * 6 * sigma, colors='k')
            ax.set_xlabel('offset (au)')
            ax.set_ylabel('velocity (km/s)')
            ax = fig.add_subplot(1, 2, 2)
            ax.contour(self.r, self.v, self.dpvminor,
                       levels=np.arange(1, 10) * 6 * sigma, colors='k')
            ax.set_xlabel('offset (au)')
            plt.show()
    
    def put_PV(self, pvmajorfits: str, pvminorfits: str, dist: float,
               vsys: float, rmax: float, vmin: float, vmax: float, sigma: float):
        self.read_pvfits(pvmajorfits, dist, vsys, rmax, vmin, vmax, sigma)
        self.dpvmajor = self.data
        self.read_pvfits(pvminorfits, dist, vsys, rmax, vmin, vmax, sigma)
        self.dpvminor = self.data
        self.r = self.x
        
    def fitting(self, incl: float, Mstar_range: list = [0, 10],
                Rc_range: list = [0, 1000], cutoff: float = 5,
                vmask: list = [0, 0],
                show: bool = False, figname: str = 'PVsilhouette'):
        
        majormap = np.where(self.dpvmajor > cutoff * self.sigma, 1, 0)
        minormap = np.where(self.dpvminor > cutoff * self.sigma, 1, 0)
        majormap[(vmask[0] < self.v) * (self.v < vmask[1])] = 1
        minormap[(vmask[0] < self.v) * (self.v < vmask[1])] = 1
        allpix = len(np.ravel(majormap)) + len(np.ravel(minormap))
        def lnprob(p):
            q = 10**p
            a = velmax(self.r, Mstar=q[0], Rc=q[1], incl=incl)
            majorvel = []
            for min, max in zip(a['major']['vlosmin'], a['major']['vlosmax']):
                majorvel.append(np.where((min < self.v) * (self.v < max), 1, 0))
            majorvel = np.transpose(majorvel)
            minorvel = []
            for min, max in zip(a['minor']['vlosmin'], a['minor']['vlosmax']):
                minorvel.append(np.where((min < self.v) * (self.v < max), 1, 0))
            minorvel = np.transpose(minorvel)
            chi2 = np.sum((majormap - majorvel)**2) + np.sum((minormap - minorvel)**2)
            chi2r = chi2 / allpix
            print([f'{q[0]:05.3f}', f'{q[1]:07.3f}'], f'{chi2r:05.3f}')
            return -0.5 * chi2
        plim = np.log10([Mstar_range, Rc_range]).T
        popt, perr = emcee_corner(plim, lnprob, args=[],
                                  nwalkers_per_ndim=16,
                                  nburnin=200, nsteps=200,
                                  labels=['log Mstar', 'log Rc'],
                                  rangelevel=0.9,
                                  figname=figname+'.corner.png',
                                  show_corner=True)
        popt = 10**popt
        perr = popt * np.log(10) * perr
        self.popt = popt
        self.perr = perr
        print(popt)
        if show:
            a = velmax(self.r, Mstar=popt[0], Rc=popt[1], incl=incl)
            fig = plt.figure()
            ax = fig.add_subplot(1, 2, 1)
            ax.contour(self.r, self.v, self.dpvmajor,
                       levels=np.arange(1, 10) * 6 * sigma, colors='k')
            ax.plot(self.r, a['major']['vlosmax'], '-r')
            ax.plot(self.r, a['major']['vlosmin'], '-r')
            ax.set_xlabel('offset (au)')
            ax.set_ylabel('velocity (km/s)')
            ax.set_ylim(self.v.min(), self.v.max())
            ax = fig.add_subplot(1, 2, 2)
            ax.contour(self.r, self.v, self.dpvminor,
                       levels=np.arange(1, 10) * 6 * sigma, colors='k')
            ax.plot(self.r, a['minor']['vlosmax'], '-r')
            ax.plot(self.r, a['minor']['vlosmin'], '-r')
            ax.set_xlabel('offset (au)')
            ax.set_ylim(self.v.min(), self.v.max())
            fig.savefig(figname + '.model.png')
            plt.show()


#####################################################################
if __name__ == '__main__':
    filehead = cubefits.replace('.fits', '')
    pvsil = PVSilhouette()
    #pvsil.get_PV(cubefits=cubefits, center=center, pa=pa,
    #             vsys=vsys, dist=dist, sigma=sigma,
    #             rmax=rmax, vmax=vmax, show=False)
    pvsil.put_PV(pvmajorfits=pvmajorfits, pvminorfits=pvminorfits,
                 dist=dist, vsys=vsys, rmax=rmax, vmin=vlim[0], vmax=vlim[1],
                 sigma=sigma)
    pvsil.fitting(incl=incl, Mstar_range=[0.01, 10.0], Rc_range=[5, 500],
                  cutoff=5, show=True, figname=filehead, vmask=vmask)
    