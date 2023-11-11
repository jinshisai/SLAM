# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Yusuke Aso
# Created Date: 2022 Jan 27
# version = alpha
# ---------------------------------------------------------------------------
"""
This script makes model channel maps from the observed mom0 by assuming 2D velocity pattern.
The main class ChannelFit can be imported to do each steps separately.

Note. FITS files with multiple beams are not supported. The dynamic range for xlim_plot and vlim_plot should be >10 for nice tick labels.
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import constants, units, wcs
from astropy.coordinates import SkyCoord
from scipy.signal import convolve
from scipy.interpolate import RegularGridInterpolator as RGI
from scipy.interpolate import interp1d
from scipy.special import erf
import warnings
from tqdm import tqdm
from utils import emcee_corner

warnings.simplefilter('ignore', RuntimeWarning)

GG = constants.G.si.value
M_sun = constants.M_sun.si.value
au = units.au.to('m')
vunit = np.sqrt(GG * M_sun / au) * 1e-3

def rot(x, y, pa):
    s = x * np.cos(pa) - y * np.sin(pa)  # along minor axis
    t = x * np.sin(pa) + y * np.cos(pa)  # along major axis
    return np.array([s, t])

def avefour(a: np.ndarray) -> np.ndarray:
    b = (a[:, 0::2, 0::2] + a[:, 0::2, 1::2] 
         + a[:, 1::2, 0::2] + a[:, 1::2, 1::2]) / 4.
    return b
    
def makemom01(d: np.ndarray, v: np.ndarray, sigma: float) -> dict:
    dmasked = np.nan_to_num(d)
    dv = np.min(v[1:] - v[:-1])
    mom0 = np.sum(d, axis=0) * dv
    sigma_mom0 = sigma * dv * np.sqrt(len(d))
    vv = np.moveaxis([[v]], 2, 0)
    dmasked[dmasked < 3 * sigma] = 0
    mom1 = np.sum(d * vv, axis=0) / np.sum(d, axis=0)
    mom2 = np.sqrt(np.sum(d * (vv - mom1)**2, axis=0), np.sum(d, axis=0))
    mom1[mom0 < 3 * sigma_mom0] = np.nan
    return {'mom0':mom0, 'mom1':mom1, 'mom2':mom2, 'sigma_mom0':sigma_mom0}
    
def clean(data: np.ndarray, beam: np.ndarray, sigma: float,
          threshold: float = 3, gain: float = 0.05) -> np.ndarray:
    shape = np.shape(data)
    cleancomponent = data * 0
    cleanresidual = data * 1
    beamarea = np.sum(beam)  # pixel/beam
    rms0, rms = 10000 * sigma, 10000 * sigma
    for i in range(1000000):
        if i == 1000000 - 1:
            print('\n1000000 iterations achived in CLEAN.')
            break
        if (peak := np.nanmax(cleanresidual)) < threshold * sigma:
            print('\nThreshold achieved in CLEAN. '
                  f'(rms={rms / sigma:.2f}sigma, '
                  f'peak={peak / sigma:.2f}sigma)')
            break
        print(f'\rCLEAN reached {peak / sigma:.2f}sigma.', end='')
        ip, jp = np.unravel_index(np.nanargmax(cleanresidual), shape)
        cc = np.zeros_like(cleanresidual)
        cc[ip, jp] = gain * peak / beamarea  # Jy/pixel
        newresidual = cleanresidual - convolve(cc, beam, mode='same')
        rms = np.sqrt(np.nanmean(newresidual**2))
        #if rms > rms0:
        #    print('RMS increased in CLEAN. '
        #          f'(rms={rms / sigma:.2f}sigma, '
        #          f'peak={peak / sigma:.2f}sigma)')
        #    break
        #else:
        #    rms0 = rms
        cleancomponent = cleancomponent + cc
        cleanresidual = newresidual
    cleancomponent = cleancomponent + cleanresidual / np.sum(beam)
    return cleancomponent, cleanresidual
        
    
class ChannelFit():

    def __init__(self, disk: bool = True, envelope: bool = True,
                 combine: bool = False, scaling: str = 'uniform'):
        self.paramkeys = ['Mstar', 'Rc', 'cs', 'h1', 'h2',
                          'pI', 'Rin', 'Ienv',
                          'xoff', 'yoff', 'voff', 'incl']
        self.disk = disk
        self.envelope = envelope
        self.combine = combine
        self.scaling = scaling

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

    def makegrid(self, cubefits: str = None,
                 pa: float = 0, incl: float = 90, dist: float = 1,
                 center: str = None, vsys: float = 0,
                 rmax: float = 1e4, vlim: tuple = (-100, 0, 0, 100),
                 sigma: float = None, nlayer: int = 4,
                 xskip: int = 1, yskip: int = 1, autoskip: bool = False,
                 gaussmargin: float = 1.6):
        if not (cubefits is None):
            self.read_cubefits(cubefits, center, dist, vsys,
                               -rmax, rmax, -rmax, rmax, None, None,
                               xskip, yskip, sigma)
            dpix = min([np.abs(self.dx), np.abs(self.dy)])
            if autoskip:
                iskip = int(self.bmin / (dpix / xskip) / 5)
                self.read_cubefits(cubefits, center, dist, vsys,
                                   -rmax, rmax, -rmax, rmax, None, None,
                                   iskip, iskip, sigma)
                dpix = min([np.abs(self.dx), np.abs(self.dy)])
                ibmin = self.bmin / dpix
                print(f'Adopt xskip={iskip:d} and yskip={iskip:d}.')
                print(f'Beam minor axis is {ibmin:.1f} pixels.')
            v = self.v
        self.incl0 = incl
        self.update_incl(incl)
        pa_rad = np.radians(pa)
        self.pa_rad = pa_rad
        self.cospa = np.cos(pa_rad)
        self.sinpa = np.sin(pa_rad)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        self.v_nanblue = v[v < vlim[0]]
        self.v_blue = v[(vlim[0] <= v) * (v <= vlim[1])]
        self.v_nanmid = v[(vlim[1] < v) * (v < vlim[2])]
        self.v_red = v[(vlim[2] <= v) * (v <= vlim[3])]
        self.v_nanred = v[vlim[3] < v]
        self.v_valid0 = np.r_[self.v_blue, self.v_red]
        self.v_valid1 = np.array([np.mean(self.v_blue), np.mean(self.v_red)])

        self.data_blue = self.data[(vlim[0] <= v) * (v <= vlim[1])]
        self.data_red = self.data[(vlim[2] <= v) * (v <= vlim[3])]
        self.data_valid0 = np.append(self.data_blue, self.data_red, axis=0) 
        self.data_valid1 = np.array([np.mean(self.data_blue, axis=0),
                                     np.mean(self.data_red, axis=0)]) 
        
        m = makemom01(self.data_valid0, self.v_valid0, sigma)
        self.mom0 = m['mom0']
        self.mom1 = m['mom1']
        self.mom2 = m['mom2']
        self.sigma_mom0 = m['sigma_mom0']
        X, Y = rot(self.X, self.Y, pa_rad)
        self.signmajor = np.sign(np.nansum(self.mom1 * Y))
        self.signminor = np.sign(np.nansum(self.mom1 * X)) * (-1)
        
        # 2d nested grid on the disk plane.
        # x and y are minor and major axis coordinates before projection.
        r_need = rmax + gaussmargin * self.bmaj
        npix = int(2 * r_need / dpix + 0.5)
        npix = int(4 * np.ceil(npix / 4))
        self.nq1 = npix // 2 - npix // 2 // 2
        self.nq3 = self.nq1 + npix // 2
        self.nlayer = nlayer  # down to dpix / 2**(nlayer-1)
        xnest = [None] * nlayer
        ynest = [None] * nlayer
        Xnest = [None] * nlayer
        Ynest = [None] * nlayer
        Rnest = [None] * nlayer
        for l in range(nlayer):
            n = npix // 2 - 0.5
            s = np.linspace(-n, n, npix) * dpix / 2**l
            X, Y = np.meshgrid(s, s)
            R = np.hypot(X, Y)
            xnest[l] = s
            ynest[l] = s
            Xnest[l] = X
            Ynest[l] = Y
            Rnest[l] = R
        self.xnest = np.array(xnest)
        self.ynest = np.array(ynest)
        self.Xnest = np.array(Xnest)
        self.Ynest = np.array(Ynest)
        self.Xnest, self.Ynest = rot(self.Xnest, self.Ynest, pa_rad)
        print('-------- nested grid --------')
        for l in range(len(xnest)):
            print(f'x, dx, npix: +/-{xnest[l][-1]:.2f},'
                  + f' {xnest[l][1]-xnest[l][0]:.2f} au,'
                  + f' {npix:d}')
        print('-----------------------------')
        
        ngauss = int(gaussmargin * self.bmaj / dpix + 0.5)  # 0.5 is for rounding
        xb = (np.arange(2 * ngauss + 1) - ngauss) * dpix
        yb = (np.arange(2 * ngauss + 1) - ngauss) * dpix
        xb, yb = rot(*np.meshgrid(xb, yb), np.radians(self.bpa))
        gaussbeam = np.exp(-(yb / self.bmaj)**2 - (xb / self.bmin)**2)
        self.pixperbeam = np.sum(gaussbeam)
        self.gaussbeam = gaussbeam
        
        n_need = int(r_need / dpix + 0.5)
        self.ineed0 = npix // 2 - n_need
        self.ineed1 = npix // 2 + n_need
        self.xneed = self.xnest[0][self.ineed0:self.ineed1]
        self.yneed = self.ynest[0][self.ineed0:self.ineed1]

        if self.scaling == 'mom0':
            c = clean(data=self.mom0, beam=self.gaussbeam,
                      sigma=self.sigma_mom0, threshold=3)
            self.cleancomponent, self.cleanresidual = c
            self.gaussbeam = self.gaussbeam[:, ::-1]
                
    def update_incl(self, incl: float):
        i = np.radians(self.incl0 + incl)
        self.sini = np.sin(i)
        self.cosi = np.cos(i)
        self.tani = np.tan(i)
        
    def update_xdisk(self, h1: float, h2: float = -1):
        x = [None] * 4
        for i, hdisk in zip([0, 2], [h1, h2]):
            if hdisk < 0:
                x1 = None
                x2 = None
            elif hdisk < 0.01:
                x1 = self.Xnest / self.cosi
                x2 = self.Xnest / self.cosi
            else:
                Xcosi = self.Xnest * self.cosi
                a = self.tani**(-2) - hdisk**2
                b = (1 + hdisk**2) * Xcosi
                c = (self.tani**2 - hdisk**2) * Xcosi**2 \
                    - hdisk**2 * self.Ynest**2
                if -1e-3 < a < 1e-3:
                    x1 = Xcosi + c / b / 2
                    x2 = None
                else:
                    zsini1 = np.full_like(self.Xnest, np.nan)
                    zsini2 = np.full_like(self.Xnest, np.nan)
                    c = (D := b**2 - a * c) >= 0
                    sqrtD = np.sqrt(D[c])
                    zsini1[c] = (b[c] + sqrtD) / a
                    zsini2[c] = (b[c] - sqrtD) / a
                    x1 = Xcosi + zsini1
                    x2 = Xcosi + zsini2
            x[i], x[i + 1] = x1, x2
        self.xdisk = x

    def update_prof(self, cs: float):
        cs_over_dv = cs / self.dv
        w = max([cs_over_dv * 2.35482, 1])  # 2.35482 ~ sqrt(8ln2)
        vmax_over_w = 2  # in the unit of max(FWHM, dv)
        w_over_d = 11
        vmax = vmax_over_w * w
        d = w / w_over_d
        n = 2 * int(vmax_over_w * w_over_d) + 1
        v = np.linspace(-vmax, vmax, n)
        if cs_over_dv < 0.01:
            p = (1 + np.sign(v + 0.5)) * (1 - np.sign(v - 0.5)) / 4
        else:
            p = erf((v + 0.5) / np.sqrt(2) / cs_over_dv) \
                - erf((v - 0.5) / np.sqrt(2) / cs_over_dv)
            p[0] = p[n - 1] = 0
        self.prof, self.prof_n, self.prof_d = p, n - 1, d

    def update_getvlos(self, Rc: float, Rin: float):
        def getvlos(x_in: np.ndarray):
            if x_in is None:
                return None
            r = np.hypot(x_in, self.Ynest)
            c = r > Rc
            vp = r**(-1/2)
            vr = r * 0
            if self.envelope:
                vp[c] = np.sqrt(Rc) / r[c]
                vr[c] = -r[c]**(-1/2) * np.sqrt(2 - Rc / r[c])
            erot = self.Ynest * self.signmajor / r
            erad = x_in * self.signminor / r
            vlos = (vp * erot + vr * erad) * self.sini * vunit
            vlos[r < Rin] = np.nan
            if not self.envelope:
                vlos[c] = np.nan
            if not self.disk:
                vlos[~c] = np.nan
            return vlos
        self.getvlos = getvlos
        
    def update_vlos(self):
        self.vlos = [self.getvlos(x) for x in self.xdisk]
    
    def get_Iunif(self, Mstar: float, Rc: float, pI: float,
                 Ienv: float, offvsys: float) -> np.ndarray:
        Iunif = 0
        for vlos_in, x_in in zip(self.vlos, self.xdisk):
            if vlos_in is None:
                continue
            vlos = vlos_in * np.sqrt(Mstar)
            v = np.subtract.outer(self.v_valid, vlos) - offvsys  # v, layer, y, x
            iv = v / self.dv / self.prof_d + self.prof_n // 2 + 0.5  # 0.5 is for rounding
            p = self.prof[iv.astype('int').clip(0, self.prof_n)]
            r = np.hypot(x_in, self.Ynest)
            p = np.where(r < Rc, p, p * Ienv) / (Ienv + 1)
            if pI != 0:
                p = p * r**(-pI)
            Iunif = Iunif + np.nan_to_num(p)
        for l in range(self.nlayer - 1, 0, -1):
            Iunif[:, l - 1, self.nq1:self.nq3, self.nq1:self.nq3] \
                = avefour(Iunif[:, l, :, :])
        Iunif = Iunif[:, 0, self.ineed0:self.ineed1, self.ineed0:self.ineed1]  # v, y, x
        return Iunif

    def rgi2d(self, xoff: float, yoff:float,
              I_in: np.ndarray) -> np.ndarray:
        Iout = [None] * len(I_in)
        for i, c in enumerate(I_in):
            interp = RGI((self.yneed, self.xneed), c, method='linear',
                         bounds_error=False, fill_value=0)
            Iout[i] = interp((self.Y - yoff, self.X - xoff))
        Iout = np.array(Iout)
        return Iout

    def get_scale(self, Iout, output: bool = False) -> np.ndarray:
        if self.scaling == 'chi2':
            gf = np.sum(Iout * self.data_valid, axis=(1, 2))
            ff = np.sum(Iout * Iout, axis=(1, 2))
        elif self.scaling == 'peak':
            gf = np.max(self.data_valid, axis=(1, 2))
            ff = np.max(Iout, axis=(1, 2))
        elif self.scaling == 'uniform':
            gf = np.full_like(self.v_valid, np.sum(Iout * self.data_valid))
            ff = np.full_like(self.v_valid, np.sum(Iout * Iout))
        scale = gf / ff
        scale[(ff == 0) + (scale < 0)] = 0
        if output:
            print(list(np.round(scale / np.max(scale), decimals=2)))
        return scale

    def peaktounity(self, I_in: np.ndarray) -> np.ndarray:
        xypeak = np.max(I_in, axis=(1, 2))
        scale = 1 / xypeak
        scale[xypeak == 0] = 0
        Iout = I_in * np.moveaxis([[scale]], 2, 0)
        return Iout

    def cubemodel(self, Mstar: float, Rc: float, cs: float,
                  h1: float, h2: float, pI: float, Rin: float, Ienv: float, 
                  xoff: float = 0, yoff: float = 0, voff: float = 0,
                  incl: float = 90,
                  convolving: bool = True, scaling: bool = True):
        if self.free['incl']:
            self.update_incl(incl)
        if self.free['cs']:
            self.update_prof(cs)
        if self.free['h1'] or self.free['h2']:
            self.update_xdisk(h1, h2)
        if self.free['Rc'] or self.free['Rin']:
            self.update_getvlos(Rc, Rin)
        if self.free['h1'] or self.free['h2'] \
            or self.free['Rc'] or self.free['Rin']:
            self.update_vlos()

        Iunif = self.get_Iunif(Mstar, Rc, pI, Ienv, voff)
        if self.scaling == 'mom0':
            Iunif = self.rgi2d(xoff, yoff, Iunif)
            mom0unif = np.sum(Iunif, axis=0) * self.dv
            mom0unif[mom0unif < 0] = np.nan
            Iunif = self.cleancomponent * Iunif / mom0unif
            Iunif = np.nan_to_num(Iunif)
        if convolving:
            Iout = convolve(Iunif, [self.gaussbeam], mode='same')
        else:
            Iout = self.peaktounity(Iunif)
        if self.scaling != 'mom0':
            Iout = self.rgi2d(xoff, yoff, Iout)
        if scaling:
            if self.scaling != 'mom0':
                scale = self.get_scale(Iout)
                Iout = Iout * np.moveaxis([[scale]], 2, 0)
        else:
            Iout = Iout / np.max(Iout)
        return Iout
                  
    def fitting(self, Mstar_range: list = [0.01, 10],
                Rc_range: list = [1, 1000],
                cs_range: list = [0.01, 1],
                h1_range: list = [0.01, 1],
                h2_range: list = [0.01, 1],
                pI_range: list = [-2, 2],
                Rin_range: list = [0, 1000],
                Ienv_range: list = [0.01, 100],
                xoff_range: list = [-100, 100],
                yoff_range: list = [-100, 100],
                voff_range: list = [-0.2, 0.2],
                incl_range: list = [-45, 45],
                Mstar_fixed: float = None,
                Rc_fixed: float = None,
                cs_fixed: float = None,
                h1_fixed: float = None,
                h2_fixed: float = None,
                pI_fixed: float = None,
                Rin_fixed: float = None,
                Ienv_fixed: float = None,
                xoff_fixed: float = None,
                yoff_fixed: float = None,
                voff_fixed: float = None,
                incl_fixed: float = None,
                filename: str = 'channelfit',
                show: bool = False,
                progressbar: bool = True,
                kwargs_emcee_corner: dict = {}):

        p_fixed = np.array([Mstar_fixed, Rc_fixed, cs_fixed,
                            h1_fixed, h2_fixed, pI_fixed,
                            Rin_fixed, Ienv_fixed,
                            xoff_fixed, yoff_fixed, voff_fixed,
                            incl_fixed])
        self.free = dict(zip(self.paramkeys, [p is None for p in p_fixed]))

        if not self.free['incl']:
            self.update_incl(incl_fixed)
        if not self.free['cs']:
            self.update_prof(cs_fixed)
        if not (self.free['h1'] or self.free['h2']):
            self.update_xdisk(h1_fixed, h2_fixed)
        if not (self.free['Rc'] or self.free['Rin']):
            self.update_getvlos(Rc_fixed, Rin_fixed)
        if not (self.free['h1'] or self.free['h2'] 
                or self.free['Rc'] or self.free['Rin']):
            self.update_vlos()
        
        if None in p_fixed:
            notfixed = p_fixed == None
            ilog = np.array([0, 1, 7])
            i = ilog[p_fixed[ilog] != None]
            p_fixed[i] = np.log10(p_fixed[i].astype('float'))
            labels = np.array(self.paramkeys).copy()
            labels[ilog] = ['log'+labels[i] for i in ilog]
            labels = labels[notfixed]
            kwargs0 = {'nwalkers_per_ndim':16, 'nburnin':200, 'nsteps':500,
                       'labels': labels, 'rangelevel':None,
                       'figname':filename+'.corner.png', 'show_corner':show}
            kw = dict(kwargs0, **kwargs_emcee_corner)
            if progressbar:
                total = kw['nwalkers_per_ndim'] * len(p_fixed[notfixed])
                total *= kw['nburnin'] + kw['nsteps'] + 2
                bar = tqdm(total=total)
                bar.set_description('Within the ranges')
            if self.combine:
                self.data_valid = self.data_valid1
                self.v_valid = self.v_valid1
                self.dv = self.dv * len(self.v_valid0) / 2
                self.sigma = self.sigma / np.sqrt(len(self.v_valid0) / 2)
            else:
                self.data_valid = self.data_valid0
                self.v_valid = self.v_valid0
            def lnprob(p):
                if progressbar:
                    bar.update(1)
                q = p_fixed.copy()
                q[notfixed] = p
                q[ilog] = 10**q[ilog]
                model = self.cubemodel(*q)
                chi2 = np.nansum((self.data_valid - model)**2) \
                       / self.sigma**2 / self.pixperbeam
                return -0.5 * chi2
            plim = np.array([np.log10(Mstar_range), np.log10(Rc_range),
                             cs_range, h1_range, h2_range,
                             pI_range, Rin_range, np.log10(Ienv_range),
                             xoff_range, yoff_range, voff_range,
                             incl_range])
            plim = plim[notfixed].T
            mcmc = emcee_corner(plim, lnprob, simpleoutput=False, **kw)
            if self.combine:
                self.data_valid = self.data_valid0
                self.v_valid = self.v_valid0
                self.dv = self.dv / (len(self.v_valid0) / 2)
                self.sigma = self.sigma * np.sqrt(len(self.v_valid0) / 2)
            def get_p(i: int):
                p = p_fixed.copy()
                p[notfixed] = mcmc[i]
                p[ilog] = 10**p[ilog]
                return p
            self.popt = get_p(0)
            self.plow = get_p(1)
            self.pmid = get_p(2)
            self.phigh = get_p(3)
        else:
            self.popt = p_fixed
            self.plow = p_fixed
            self.pmid = p_fixed
            self.phigh = p_fixed
        plist = [self.popt, self.plow, self.pmid, self.phigh]
        print('------------------------')
        print(f'popt :', ', '.join([f'{p:.2e}' for p in self.popt]))
        print('------------------------')
        np.savetxt(filename+'.popt.txt', plist)
        self.popt = dict(zip(self.paramkeys, self.popt))
        self.plow = dict(zip(self.paramkeys, self.plow))
        self.pmid = dict(zip(self.paramkeys, self.pmid))
        self.phigh = dict(zip(self.paramkeys, self.phigh))
 
    def modeltofits(self, filehead: str = 'best', **kwargs):
        w = wcs.WCS(naxis=3)
        h = self.header
        h['NAXIS1'] = len(self.x)
        h['NAXIS2'] = len(self.y)
        h['NAXIS3'] = len(self.v)
        h['CRPIX1'] = h['CRPIX1'] - self.offpix[0]
        h['CRPIX2'] = h['CRPIX2'] - self.offpix[1]
        h['CRPIX3'] = h['CRPIX3'] - self.offpix[2]
        nx = h['NAXIS1']
        ny = h['NAXIS2']
        for k in self.free.keys():
            self.free[k] = True
        if kwargs != {}:
            self.popt = kwargs
        self.data_valid = self.data_valid0
        self.v_valid = self.v_valid0
        m = self.cubemodel(**self.popt)
        m0 = self.cubemodel(**self.popt, convolving=False, scaling=False)
        m1 = self.cubemodel(**self.popt, scaling=False)
        
        def concat(m):
            if len(self.v_red) > 0:
                m_blue = m[self.v_valid < np.min(self.v_red)]
            else:
                m_blue = m * 1
                m_red = np.full((0, ny, nx), np.nan)
            if len(self.v_blue) > 0:
                m_red = m[np.max(self.v_blue) < self.v_valid]
            else:
                m_red = m * 1
                m_blue = np.full((0, ny, nx), np.nan)
            nanblue = np.full((len(self.v_nanblue), ny, nx), np.nan)
            nanmid = np.full((len(self.v_nanmid), ny, nx), np.nan)
            nanred = np.full((len(self.v_nanred), ny, nx), np.nan)
            model = nanblue
            if len(m_blue) > 0:
                model = np.append(model, m_blue, axis=0)
            if len(nanmid) > 0:
                model = np.append(model, nanmid, axis=0)
            if len(m_red) > 0:
                model = np.append(model, m_red, axis=0)
            if len(nanred) > 0:
                model = np.append(model, nanred, axis=0)
            return model
                
        def tofits(d: np.ndarray, ext: str):
            header = w.to_header()
            hdu = fits.PrimaryHDU(d, header=header)
            for k in h.keys():
                if not ('COMMENT' in k or 'HISTORY' in k):
                    hdu.header[k]=h[k]
            hdu = fits.HDUList([hdu])
            hdu.writeto(f'{filehead}.{ext}.fits', overwrite=True)
            
        tofits((model := concat(m)), 'model')
        tofits(self.data - model, 'residual')
        tofits(concat(m0), 'beforeconvolving')
        tofits(concat(m1), 'beforescaling')
        
    def plotmom(self, mode: str, filename: str = 'mom01.png', **kwargs):
        if 'mod' in mode or 'res' in mode or 'clean' in mode:
            if kwargs != {}:
                self.popt = kwargs
            d = self.cubemodel(**self.popt)
            m = makemom01(d, self.v_valid0, self.sigma)
        if 'obs' in mode:
            mom0 = self.mom0
            mom1 = self.mom1
            label = 'Obs.'
        elif 'mod' in mode:
            mom0 = m['mom0']
            mom1 = m['mom1']
            label = 'Model'
        elif 'res' in mode:
            mom0 = self.mom0 - m['mom0']
            mom1 = self.mom1 - m['mom1']
            label = r'Obs. $-$ model'
        levels = 6 * self.sigma_mom0
        levels = np.arange(1, 20) * levels
        levels = np.sort(np.r_[-levels, levels])
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        vplot = (np.nanpercentile(self.mom1, 95) 
                 - np.nanpercentile(self.mom1, 5)) / 2.
        m = ax.pcolormesh(self.x, self.y, mom1, cmap='jet',
                          vmin=-vplot, vmax=vplot)
        fig.colorbar(m, ax=ax, label=label + r' mom1 (km s$^{-1}$)')
        ax.contour(self.x, self.y, mom0, colors='gray', levels=levels)
        r = np.linspace(-1, 1, 3) * self.x.max() * 1.42
        ax.plot(r * self.sinpa, r * self.cospa, 'k:')
        ax.plot(r * self.cospa, -r * self.sinpa, 'k:')
        ax.set_xlabel('R.A. offset (au)')
        ax.set_ylabel('Dec. offset (au)')
        ax.set_xlim(self.x.max() * 1.01, self.x.min() * 1.01)
        ax.set_ylim(self.y.min() * 1.01, self.y.max() * 1.01)
        ax.set_aspect(1)
        fig.savefig(filename)
        plt.close()

    def plotclean(self, filename: str = 'clean.png'):
        cc = self.cleancomponent / self.sigma_mom0
        cr = self.cleanresidual / self.sigma_mom0
        for c, vmin, vmax, s in zip([cc, cr], [cc.min(), -3], [cc.max(), 3],
                                    ['component', 'residual']):
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            m = ax.pcolormesh(self.x, self.y, c, cmap='jet', vmin=vmin, vmax=vmax)
            fig.colorbar(m, ax=ax, label=f'CLEAN {s} / ' + r'$\sigma$')
            r = np.linspace(-1, 1, 3) * self.x.max() * 1.42
            ax.plot(r * self.sinpa, r * self.cospa, 'k:')
            ax.plot(r * self.cospa, -r * self.sinpa, 'k:')
            ax.set_xlabel('R.A. offset (au)')
            ax.set_ylabel('Dec. offset (au)')
            ax.set_xlim(self.x.max() * 1.01, self.x.min() * 1.01)
            ax.set_ylim(self.y.min() * 1.01, self.y.max() * 1.01)
            ax.set_aspect(1)
            fig.savefig(filename.replace('.png', '') + f'.clean{s}.png')
            plt.close()

    def equivelocity(self, filename: str = 'equivel.png', **kwargs):
        if kwargs != {}:
            self.popt = kwargs
        m = self.popt['Mstar']
        h1 = self.popt['h1']
        h2 = self.popt['h2']
        if h1 < 0: h1 = np.nan
        if h2 < 0: h2 = np.nan
        incl = self.incl0 + self.popt['incl']
        irad = np.radians(incl)
        sini = np.sin(irad)
        cosi = np.cos(irad)
        def getxy(ymax):
            y = np.linspace(-ymax, ymax, 128)
            v, y = np.meshgrid(self.v_valid, y)
            y[y * np.sign(v) < 0] = np.nan
            r = (m / (v / sini / vunit / y)**2)**(1/3)
            x = np.sqrt(r**2 - y**2)
            return x, y, r
        x, y, _ = getxy(np.max(np.abs(self.x)))
        ymax = np.max(np.abs(y[~np.isnan(x)]))
        x, y, r = getxy(ymax)
        xcosi = x * cosi
        rsini = r * sini
        x = [xcosi + h1 * rsini,
             xcosi - h1 * rsini,
             xcosi + h2 * rsini,
             xcosi - h2 * rsini,
             -xcosi + h1 * rsini,
             -xcosi - h1 * rsini,
             -xcosi + h2 * rsini,
             -xcosi - h2 * rsini,
        ]
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        for i in range(8):
            for xx, yy in zip(x[i], y):
                m = ax.scatter(xx, yy, c=self.v_valid, cmap='jet', s=1,
                               vmin=self.v.min(), vmax=self.v.max())
        fig.colorbar(m, ax=ax, label=r'$V_{\rm los}$ (km s$^{-1}$)')
        ax.set_xlabel('major offset (au)')
        ax.set_ylabel('minor offset (au)')
        ax.set_xlim(-ymax * 1.01, ymax * 1.01)
        ax.set_ylim(-ymax * 1.01, ymax * 1.01)
        ax.set_aspect(1)
        fig.savefig(filename)
        plt.close()
        
