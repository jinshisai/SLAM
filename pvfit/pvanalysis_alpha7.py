# -*- coding: utf-8 -*-
#----------------------------------------------------------------------------
# Created By  : Yusuke Aso
# Created Date: 2022 Jan 28
# version = alpha6
# ---------------------------------------------------------------------------
"""
This script derives the edge and ridge points from a position-velocity (PV) diagram, in the FITS form (AXIS1=arcsec, AXIS2=Hz) and fits the points with a double-power-law function. The outputs are linear and log-log PV diagrams and corner plots derived from the MCMC fitting.
The main class PVAnalysis can be imported to do each steps separately: get edge/ridge, write them, fit them, output the fit result, and plot PV diagrams.

Note. FITS files with multiple beams are not supported. A part of this script is different for emcee 2.2.1 and 3.1.1. The dynamic range for xlim_plot and vlim_plot should be >10 for nice tick labels.
"""

############ INPUTS ############
pvfits   = '../testfits/TMC1A_C18O_t2000klam.image.pv_invert.fits'
incl     = 48.  # deg
vsys     = 6.4  # km/s
dist     = 140.  # pc
sigma    = 1.7e-3  # Jy/beam; None means an automatic estimation.
cutoff   = 5.0  # sigma
quadrant = '24'  # '13' or '24', in which quadrants emission is.
Mlim = (0.0, 10.0)  # M_sun; to remove unreasonable points.
xlim = (0, 200)  # au
vlim = (0, 5.0)  # km/s
xlim_plot = (200 / 20, 200)  # au
vlim_plot = (6.0 / 20, 6.0)  # km/s
use_velocity = True  # False: representative velocities are not used.
use_position = True  # False: representative positions are not used.
include_vsys = False  # False: vsys is fixed at the value above.
include_dp   = True  # False: dp=0, i.e., single power.
include_pin  = False  # False: pin=0.5, i.e., Keplerian.
show_pv      = True  # True: plot PV and log-log diagrams.
show_corner  = True  # True: plot the triangle diagram of the MCMC fitting.
write_point  = False  # True: write edge/ridge points to a dat file.
minrelerr = 0.01  # minimum of relative error.
minabserr = 0.1  # minimum of absolute error in the units of bmaj and dv.
ridge = 'mean'  # mean or gauss
interp_ridge = False  # True: ridge is derived after interpolation.
################################


import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy import constants, units
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
from scipy.interpolate import RectBivariateSpline as RBS
import emcee, corner, sys

from analysis_tools import edge, ridge_gauss, ridge_mean, doublepower_v, doublepower_v_error, doublepower_r, doublepower_r_error
sys.path.append('../')
from utils import emcee_corner


def kepler_mass(r, v, unit):
    return v**2 * np.abs(r) * unit

def kepler_mass_error(r, v, dr, dv, unit):
    return kepler_mass(r, v, unit) * np.sqrt((2 * dv / v)**2 + (dr / r)**2)

def between(t, tlim):
    return (tlim[0] < t) * (t < tlim[1])

def flipx(Ds):
    return [Ds[0], -Ds[1], Ds[2], -Ds[3], Ds[4], Ds[5]]



class PVAnalysis():
    """
    self.x : 1D numpy array
        Positional coordinate of the PV diagram.
    self.dx : float
        The increment of x.
    self.v : 1D numpy array
        Velocity coordinate of the PV diagram.
    self.dv : float
        The increment of v.
    self.data : 2D numpy array
        Intensities of the PV diagram.
    self.header : the header of astropy.io.fits
        Header of the input FITS file.
    self.sigma : float
        Standard deviation of the PV diagram.
    self.bmaj : float
        Major axis of the beam.
    self.dNyquist : float
        bmaj * 0.5 if there is a beam else dx.
    self.edgeridge : dict
        {'edge':{'position':[v, x, dx], 'velocity':[x, v, dv]}, 'ridge':{...}}
    self.popt : dict
        {'edge':[[rb, vb, pin, dp, vsys], [drb, dvb, dpin, ddp, dvsys]], 'ridge'[[...], [...]]}
    self.rvlim : dict
        {'edge':[[rin, rout], [vin, vout]], 'ridge':[[...], [...]]}
    """

#    def __init__(self):

    def read_pvfits(self, pvfits, dist, vsys=0, xmax=1e4, vmax=100, sigma=None):
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
        j0, j1 = np.argmin(np.abs(v + vmax)), np.argmin(np.abs(v - vmax))
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


    def get_edgeridge(self, pvfits=None, dist=None, vsys=None, sigma=None,
                      incl=90., cutoff=5, quadrant='13',
                      Mlim=(0., 10.), xlim=(0., 1000.), vlim=(0., 100.),
                      use_velocity=True, use_position=True,
                      minrelerr=0.01, minabserr=0.1,
                      ridge='mean', interp_ridge=False):
        """
        Get the edge/ridge positions/velocities.

        Parameters
        ----------
        pvfits : str or None
            Name of the input FITS file including the extension. None means the input (pvfits, dist, vsys, sigma) for read_pvfits is used if read_pvfits was executed.
        dist : float
            Distance of the target, used to convert arcsec to au.
        vsys : float
            Systemic velocity of the target.
        sigma : float or None
            Standard deviation of the FITS data. None means automatic.
        incl : float
            Inclination angle of the target in the unit of degree, used to calculate the central stellar mass from a radius and a velocity.
        cutoff : float
            The edge level is cutoff * sigma. The ridge also uses the data above cutoff * sigma.
        quadrant : '13' of '24'
            Quadrant of the PV diagram where the emission is expected.
        Mlim : tuple of float
            Data points outside Mlim are ignored.
        xlim : tuple of float
            The edge/ridge points are derived in this x range.
        vlim : tuple of float
            The edge/ridge points are derived in this v range.
        use_velocity : bool
            False does not use edge/ridge velocities.
        use_position : bool
            False does not use edge/ridge positions.
        minrelerr : float
            Relative error bars for the edge/ridge points are limited above this value.
        minabserr : float
            Absolute error bars for the edge/ridge points are limited above this value. The unit is the beam major axis or the vleocity resolution.
        ridge : 'mean' or 'gauss'
            The definition of ridge positions/velocities. The intensity weighted mean or the gaussian center.
        interp_ridge : bool
            True means the ridge positions/velocities are derived after the PV diagram is interpolated.

        Returns
        ----------
        fit_result : dict
            {'edge':{'position':[...], 'velocity':[]} 'ridge':{...}}
            'position' is a list of [v, x, dx] where x is a function of v.
            'velocity' is a list of [x, v, dv] where v is a function of x.
        """
        if not (pvfits is None):
            self.read_pvfits(pvfits, dist, vsys, xlim[1], vlim[1], sigma)
        x, dx, dNyquist = self.x, self.dx, self.dNyquist
        v, dv = self.v, self.dv
        sigma = self.sigma
        d = self.data[:, ::-1] if quadrant == '24' else self.data

        GG = constants.G.si.value
        M_sun = constants.M_sun.si.value
        au = units.au.to('m')
        self.__unit = 1.e6 * au / GG / M_sun / np.sin(np.radians(incl))**2

        thre = cutoff * sigma    
        def clipped_error(err, val, mode):
            res = self.bmaj if mode == 'x' else dv
            return max(err, minrelerr * np.abs(val), minabserr * res)
    
        ### Get edge/ridge positions and velocities. ###
        def get_points(mode):
            if mode == 'x':
                s, ds, t, slim, tlim, d_input = x, dx, v, xlim, vlim, d
            elif mode == 'v':
                s, ds, t, slim, tlim, d_input = v, dv, x, vlim, xlim, d.T
            s_e, ds_e, s_r, ds_r = [], [], [], []
            si_org = np.linspace(s[0], s[-1], (len(s) - 1) * 10 + 1)
            for tt, dd in zip(t, d_input):
                tt_s, tt_a = np.sign(tt), np.abs(tt)
                if not between(tt_a, tlim) \
                    or (mode == 'v' and not use_velocity) \
                    or (mode == 'x' and not use_position):
                    for a in [s_e, ds_e, s_r, ds_r]: a.append(np.nan)
                    continue
                # Edge
                if mode == 'x':
                    M = kepler_mass(si_org, tt, self.__unit)
                elif mode == 'v':
                    M = kepler_mass(tt, si_org, self.__unit)
                d0 = interp1d(s, dd, kind='cubic')(si_org)
                val, err = edge(si_org, d0, sigma, thre, between(M, Mlim), tt)
                err = clipped_error(err, val, mode=mode)
                if not between(val * tt_s, slim): val = np.nan
                for a, b in zip([s_e, ds_e], [val, err]): a.append(b)
                # Ridge
                if interp_ridge:
                    d1, ss = interp1d(s, dd, kind='cubic')(si_org), si_org
                else:
                    d1, ss = dd, s
                cond = (d1 > thre)
                d1, ss = d1[cond], ss[cond]
                if ridge == 'mean':
                    val, err = ridge_mean(ss, d1, sigma)
                elif ridge == 'gauss':
                    val, err = ridge_gauss(ss, d1, sigma)
                err = clipped_error(err, val, mode=mode)
                if not between(val * tt_s, slim): val = np.nan
                for a, b in zip([s_r, ds_r], [val, err]): a.append(b)
            return np.array([s_e, ds_e, s_r, ds_r])
        x_e, dx_e, x_r, dx_r = get_points(mode='x')
        v_e, dv_e, v_r, dv_r = get_points(mode='v')

        ### Sample edge/ridge velocities at the Nyquist rate. ###
        sNyquist = (iNyquist := int(round(dNyquist / dx))) // 2
        sampleN = lambda a: np.array(a[sNyquist::iNyquist])
        v_e,dv_e,v_r,dv_r,xN = list(map(sampleN, [v_e,dv_e,v_r,dv_r,x]))
        
        ### Arrange red/blue points separately without their sign. ###
        rmsign = lambda a: [a[v > 0], -a[::-1][v[::-1] < 0]]
        x1_e,dx1_e,x1_r,dx1_r,v0 = list(map(rmsign, [x_e,dx_e,x_r,dx_r,v]))
        rmsign = lambda a: [a[xN > 0], -a[::-1][xN[::-1] < 0]]
        v1_e,dv1_e,v1_r,dv1_r,x0 = list(map(rmsign, [v_e,dv_e,v_r,dv_r,xN]))
        v0_e, v0_r, x0_e, x0_r = v0.copy(), v0.copy(), x0.copy(), x0.copy()
        Es = [v0_e, x1_e, dx1_e, x0_e, v1_e, dv1_e]
        Rs = [v0_r, x1_r, dx1_r, x0_r, v1_r, dv1_r]
        
        ### Remove points after turn over. ###
        for a in [Es[1], Es[4], Rs[1], Rs[4]]:
            for i in range(2):
                if np.any(~np.isnan(a[i])):
                    a[i][:np.nanargmax(a[i])] = np.nan
    
        ### Remove points outside Mlim. ###
        for Ds in [Es, Rs]:
            for i in range(2):
                M = kepler_mass(Ds[1][i], Ds[0][i], self.__unit)
                Ds[1][i][~between(M, Mlim)] = np.nan
                M = kepler_mass(Ds[3][i], Ds[4][i], self.__unit)
                Ds[4][i][~between(M, Mlim)] = np.nan
    
        ### Remove low-velocity positions and inner velocities. ###
        for Ds in [Es, Rs]:
            for i in range(2):
                xx, xv = np.meshgrid(Ds[1][i], Ds[3][i])
                vx, vv = np.meshgrid(Ds[0][i], Ds[4][i])
                xvd = np.hypot((xx - xv) / dNyquist, (vx - vv) / dv)
                if np.any(~np.isnan(xvd)):
                    iv, ix = np.unravel_index(np.nanargmin(xvd), np.shape(xx))
                    Ds[1][i][:ix] = np.nan
                    Ds[4][i][:iv] = np.nan
        
        ### Combine blue/red, remove nan, and sort. ###
        def comb_rmnan_sort(Dsin):
            Dsout = [[], [], [], [], [], []]
            PVin, PVout = [Dsin[:3], Dsin[3:]], [Dsout[:3], Dsout[3:]]
            for c, A, B in zip([Dsin[1], Dsin[4]], PVout, PVin):
                for i, s in zip([0, 1], [1, -1]):
                    for a, b in zip(A, B):
                        a.extend(s * b[i][~np.isnan(c[i])])
            s = [np.argsort(Dsout[0])] * 3 + [np.argsort(Dsout[3])] * 3
            for i in range(6): Dsout[i] = np.array(Dsout[i])[s[i]]
            return Dsout
        Es = comb_rmnan_sort(Es)
        Rs = comb_rmnan_sort(Rs)
        if quadrant == '24': Es, Rs = flipx(Es), flipx(Rs)
        result = {'edge':{'position':Es[:3], 'velocity':Es[3:]},
                  'ridge':{'position':Rs[:3], 'velocity':Rs[3:]}}
        self.edgeridge = result
        self.__Es = result['edge']['position'] + result['edge']['velocity']
        self.__Rs = result['ridge']['position'] + result['ridge']['velocity']
        self.quadrant = quadrant
        return result


    def write_edgeridge(self, filehead='pvanalysis'):
        """
        Write the edge/ridge positions/velocities to a text file.

        Parameters
        ----------
        filehead : str
            The output text file has a name of "filehead".edge.dat. and "filehead".ridge.dat The file consists of four columns of x (au), dx (au), v (km/s), and dv (km/s).
        """
        for title, As in zip(['edge', 'ridge'], [self.__Es, self.__Rs]):
            v0, x1, dx1, x0, v1, dv1 = As
            np.savetxt(filehead + '.' + title + '.dat',
                       np.c_[np.r_[x1, x0], np.r_[dx1, x0 * 0],
                             np.r_[v0, v1], np.r_[v0 * 0, dv1]],
                       header='x (au), dx (au), v (km/s), dv (km/s)')
        print(f'- Wrote to {filehead}.edge.dat and {filehead}.ridge.dat.')


    def fit_edgeridge(self, include_vsys=False, include_dp=True,
                      include_pin=False, filehead='pvanalysis',
                      show_corner=False):
        """
        Fit the derived edge/ridge positions/velocities with a double power law function by using emcee.

        Parameters
        ----------
        include_vsys : bool
            False means vsys is fixed at 0.
        include_dp : bool
            False means dp is fixed at 0, i.e., single power law function.
        include_pin : bool
            False means pin is fixed at 0.5, i.e., the Keplerian law.
        filehead : str
            The output corner figures have names of "filehead".corner_e.png and "filehead".corner_r.png.
        show_corner : bool
            True means the corner figures are shown. These figures are also plotted in two png files.

        Returns
        ----------
        fitsdata : dict
            {'edge':{'popt':[...], 'perr':[...]}, 'ridge':{...}}
            'popt' is a list of [r_break, v_break, p_in, dp, vsys].
            'perr' is the uncertainties for popt.
        """
        self.include_dp = include_dp
        labels = np.array(['Rb', 'Vb', 'p_in', 'dp', 'Vsys'])
        include = [True, include_dp, include_pin, include_dp, include_vsys]
        labels = labels[include]
        Es = flipx(self.__Es) if self.quadrant == '24' else self.__Es
        Rs = flipx(self.__Rs) if self.quadrant == '24' else self.__Rs
        popt_e, popt_r = [np.empty(5), np.empty(5)], [np.empty(5), np.empty(5)]
        for args, ext, res in zip([Es, Rs], ['_e', '_r'], [popt_e, popt_r]):
            plim = np.array([
                   [np.min(np.abs(np.r_[args[1], args[3]])),
                    np.min(np.abs(np.r_[args[0], args[4]])),
                    0.01,
                    0.,
                    -1],
                   [np.max(np.abs(np.r_[args[1], args[3]])),
                    np.max(np.abs(np.r_[args[0], args[4]])),
                    10.,
                    10,
                    1.]])
            q0 = np.array([0, np.sqrt(plim[0][1] * plim[1][1]), 0.5, 0, 0])
            q0 = np.where(include, np.nan, q0)
            def wpow_r_custom(v, *p):
                (q := q0 * 1)[np.isnan(q0)] = p
                return doublepower_r(v, *q)
            def wpow_v_custom(r, *p):
                (q := q0 * 1)[np.isnan(q0)] = p
                return doublepower_v(r, *q)
            def lnprob(p, v0, x1, dx1, x0, v1, dv1):
                chi2 = np.sum(((x1 - wpow_r_custom(v0, *p)) / dx1)**2) \
                       + np.sum(((v1 - wpow_v_custom(x0, *p)) / dv1)**2)
                return -0.5 * chi2
            plim = plim[:, include]
            popt, perr = emcee_corner(plim, lnprob, args=args,
                                      labels=labels,
                                      figname=filehead+'.corner'+ext+'.png',
                                      show_corner=show_corner,
                                      ndata=len(args[0]) + len(args[3]))
            (qopt := q0 * 1)[np.isnan(q0)] = popt
            (qerr := q0 * 0)[np.isnan(q0)] = perr
            res[:] = [qopt, qerr]
        print(f'- Plotted in {filehead}.corner_e.png '
              + f'and {filehead}.corner_r.png')
        self.popt = {'edge':popt_e, 'ridge':popt_r}
        result = {'edge':{'popt':popt_e[0], 'perr':popt_e[1]},
                  'ridge':{'popt':popt_r[0], 'perr':popt_r[1]}}
        return result


    def get_range(self):
        """
        Calculate the ranges of the edge/ridge positions (radii) and velocities.

        Parameters
        ----------
        No parameter.

        Returns
        ----------
        fitsdata : dict
            {'edge':{'rlim':[...], 'vlim':[...]}, 'ridge':{...}}
            'rlim' is a list of [r_minimum, r_maximum].
            'vlim' is a list of [v_minimum, v_maximum].
        """
        def inout(v0, x1, dx1, x0, v1, dv1, popt):
            xall, vall = np.abs(np.r_[x0, x1]), np.abs(np.r_[v0, v1])
            rin, rout = np.min(xall), np.max(xall)
            vin, vout = np.max(vall), np.min(vall)
            rin  = doublepower_r(vin, *popt)
            if use_velocity:
                vout = doublepower_v(rout, *popt)
            else:
                rout = doublepower_r(vout, *popt)
            return [[rin, rout], [vin, vout]]
        lims_e = inout(*self.__Es, self.popt['edge'][0])
        lims_r = inout(*self.__Rs, self.popt['ridge'][0])
        self.rvlim = {'edge':lims_e, 'ridge':lims_r}
        result = {'edge':{'rlim':lims_e[0], 'vlim':lims_e[1]},
                  'ridge':{'rlim':lims_r[0], 'vlim':lims_r[1]}}
        return result


    def output_fitresult(self):
        """
        Output the fitting result in the terminal.

        Parameters
        ----------
        No parameter.

        Returns
        ----------
        No return.
        """
        if not hasattr(self, 'rvlim'): self.get_range()
        for i in ['edge', 'ridge']:
            rvlim, popt = self.rvlim[i], self.popt[i]
            rin, rout, vin, vout = *rvlim[0], *rvlim[1]
            print('--- ' + i.title() + ' ---')
            rb, vb, pin, dp, vsys = popt[0]
            drb, dvb, dpin, ddp, dvsys = popt[1]
            params = [*popt[0], *popt[1]]
            print(f'R_b   = {rb:.2f} +/- {drb:.2f} au')
            print(f'V_b   = {vb:.3f} +/- {dvb:.3f} km/s')
            print(f'p_in  = {pin:.3f} +/- {dpin:.3f}')
            print(f'dp    = {dp:.3f} +/- {ddp:.3f}')
            print(f'v_sys = {vsys:.3f} +/- {dvsys:.3f}')
            print(f'r     = {rin:.2f} --- {rout:.2f} au')
            print(f'v     = {vout:.3f} --- {vin:.3f} km/s')
            M_in = kepler_mass(rin, vin, self.__unit)
            M_b = kepler_mass(rb, vb, self.__unit)
            M_out = kepler_mass(rout, vout, self.__unit)
            drin = doublepower_r_error(vin, *params)
            dM_in = M_in * drin / rin
            if use_velocity:
                dvout = doublepower_v_error(rout, *params)
                dM_out = 2. * M_out * dvout / vout
            else:
                drout = doublepower_r_error(vout, *params)
                dM_out = M_out * drout / rout
            dM_b = kepler_mass_error(rb, vb, drb, dvb, self.__unit)
            print(f'M_in  = {M_in:.3f} +/- {dM_in:.3f} Msun')
            print(f'M_out = {M_out:.3f} +/- {dM_out:.3f} Msun')
            print(f'M_b   = {M_b:.3f} +/- {dM_b:.3f} Msun')


    def plot_edgeridge(self, xlim=(5, 500), vlim=(0.1, 10),
                       filehead='pvanalysis', show_pv=False):
        """
        Make figures of linear and log-log PV diagrams with the edge/ridge positions and velocities.

        Parameters
        ----------
        xlim : tuple of float
            The positional (radius) axis of the figure is limited to this range.
        vlim : tuple of float
            The velocity axis of the figure is limited to this range.
        filehead : str
            The figures have names of "filehead".pvlin.png and "filehead".pvlog.png.
        show_pv : bool
            True means the two figures are shown. These figures are also plotted in two png files.

        Returns
        ----------
        No return.
        """
        if np.abs(self.x).max() < xlim[1] or np.abs(self.v).max() < vlim[1]:
            self.read_pvfits(self.pvfits, self.dist, self.vsys,
                             xlim[1], vlim[1], self.sigma)
        x, dx, v, dv = self.x, self.dx, self.v, self.dv
        d, sigma = self.data, self.sigma

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

        Es, Rs = self.__Es, self.__Rs
        for Ds in [Es, Rs]:
            Ds[1] = [Ds[1][Ds[0] > 0], Ds[1][Ds[0] < 0]]
            Ds[2] = [Ds[2][Ds[0] > 0], Ds[2][Ds[0] < 0]]
            Ds[0] = [Ds[0][Ds[0] > 0], Ds[0][Ds[0] < 0]]
            Ds[3] = [Ds[3][Ds[4] > 0], Ds[3][Ds[4] < 0]]
            Ds[5] = [Ds[5][Ds[4] > 0], Ds[5][Ds[4] < 0]]
            Ds[4] = [Ds[4][Ds[4] > 0], Ds[4][Ds[4] < 0]]
        
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        # PV diagram
        pco = ax.pcolormesh(x, v, np.log10(d.clip(sigma, None)),
                            cmap='viridis', shading='nearest',
                            vmin=np.log10(sigma))
        ax.contour(x, v, d, [3 * sigma, 6 * sigma], colors='lime')
        cb = plt.colorbar(pco, ax=ax, label=r'log Jy beam$^{-1}$')
        cb.set_ticks(ticks := cb.get_ticks())
        cb.set_ticklabels([rf'${t:.1f}$' for t in ticks])
        q = -1 if self.quadrant == '24' else 1
        # Model curves
        for i, ls in zip(['edge', 'ridge'], ['--', '-']):
            popt = self.popt[i][0]
            rb, vb, vsys = popt[[0, 1, 4]]
            s = np.linspace(*self.rvlim[i][0], 100)
            s = np.r_[-s[::-1], np.nan, s]
            ax.plot(q * s, doublepower_v(s, *popt), ls, color='gray', lw=2)
            if self.include_dp:
                for s in [rb, -rb]: ax.axvline(q * s, color='k', lw=2, ls=ls) 
                for t in [vb, -vb]: ax.axhline(t+vsys, color='k', lw=2, ls=ls)
        # Edge/ridge points
        for As, fmt in zip([Es, Rs], ['v', 'o']):
            v0, x1, dx1, x0, v1, dv1 = As
            for i, c in zip([0, 1], ['pink', 'cyan']):
                ax.plot(x0[i], v1[i], fmt, color=c, ms=5)
            for i, c in zip([0, 1], ['red',  'blue']):
                ax.plot(x1[i], v0[i], fmt, color=c, ms=5)
        ax.set_xlim(-xlim_plot[1], xlim_plot[1])  # au
        ax.set_ylim(-vlim_plot[1], vlim_plot[1])  # km/s
        ax.set_xlabel('Position (au)')
        ax.set_ylabel(r'Velocity (km s$^{-1}$)')
        fig.tight_layout()
        fig.savefig(filehead + '.pvlin.png', transparent=True)
        if show_pv: plt.show()
        plt.close()
    
        ### Make mean PV diagram for logR-logV diagram. ###
        m_i, n_i = int(xlim_plot[1] / dx), int(vlim_plot[1] / dv)
        x_i = np.linspace(-m_i * dx, m_i * dx, 2 * m_i + 1)
        v_i = np.linspace(-n_i * dv, n_i * dv, 2 * n_i + 1)
        d = self.data[:, ::-1] if self.quadrant == '24' else self.data
        vsys = self.popt['edge'][0][4] * 0.5 + self.popt['ridge'][0][4] * 0.5
        f = RBS(v - vsys, x, d)
        d_i = f(v_i, x_i)
        d = (d_i + d_i[::-1, ::-1]) / 2.
        x_i, v_i, d = x_i[m_i:], v_i[n_i:], d[n_i:, m_i:]
        
        ### Make logR-logV diagram. ###
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(1, 1, 1)
        # Log-log PV diagram
        pco = ax.pcolormesh(x_i, v_i, np.log10(d.clip(sigma, None)),
                            cmap='viridis', shading='nearest',
                            vmin=np.log10(sigma))
        ax.contour(x_i, v_i, d, [3 * sigma, 6 * sigma], colors='lime')
        cb = plt.colorbar(pco, ax=ax, label=r'log Jy beam$^{-1}$')
        cb.set_ticks(ticks := cb.get_ticks())
        cb.set_ticklabels([rf'${t:.1f}$' for t in ticks])
        # Model lines
        for i, ls in zip(['edge', 'ridge'], ['--', '-']):
            rb, vb = (popt := self.popt[i][0])[[0, 1]]
            s = np.geomspace(*self.rvlim[i][0], 100)
            t = doublepower_v(s, *popt) - popt[4]
            ax.plot(s, t, ls, color='gray', lw=2)
            if self.include_dp:
                ax.axhline(vb, color='k', lw=2, ls=ls)
                ax.axvline(rb, color='k', lw=2, ls=ls)
        # Edge/ridge points
        for As, j, fmt in zip([Es, Rs], ['edge', 'ridge'], ['v', 'o']):
            v0, x1, dx1, x0, v1, dv1 = As
            vsys = self.popt[j][0][4]
            for i, w, c in zip([0, 1], [vsys, -vsys], ['pink', 'cyan']):
                ax.errorbar(np.abs(x0[i]), np.abs(v1[i]) - w,
                            yerr=dv1[i], fmt=fmt, color=c, ms=5)
            for i, w, c in zip([0, 1], [vsys, -vsys], ['red',  'blue']):
                ax.errorbar(np.abs(x1[i]), np.abs(v0[i]) - w,
                            xerr=dx1[i], fmt=fmt, color=c, ms=5)
        ax.set_xscale('log')
        ax.set_yscale('log')
        def nice_ticks(ticks, tlim):
            order = 10**np.floor(np.log10(tlow := tlim[0]))
            tlow = np.ceil(tlow / order) * order
            order = 10**np.floor(np.log10(tup := tlim[1]))
            tup = np.floor(tup / order) * order
            return np.sort(np.r_[ticks, tlow, tup])
        xticks = nice_ticks(ax.get_xticks(), xlim_plot)
        yticks = nice_ticks(ax.get_yticks(), vlim_plot)
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        def nice_labels(ticks):
            digits = np.floor(np.log10(ticks)).astype('int').clip(None, 0)
            return [f'{t:.{d:d}f}' for t, d in zip(ticks, -digits)]
        ax.set_xticklabels(nice_labels(xticks))
        ax.set_yticklabels(nice_labels(yticks))
        ax.set_xlim(xlim_plot[0] * 0.999, xlim_plot[1] * 1.001)  # au
        ax.set_ylim(vlim_plot[0] * 0.999, vlim_plot[1] * 1.001)  # km/s
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlabel('Radius (au)')
        ax.set_ylabel(r'Velocity (km s$^{-1}$)')
        fig.tight_layout()
        fig.savefig(filehead + '.pvlog.png', transparent=True)
        if show_pv: plt.show()
        plt.close()
        print(f'- Plotted in {filehead}.pvlin.png '
              + f'and {filehead}.pvlog.png.')

#########################################################################
if __name__ == '__main__':
    filehead = pvfits.split('/')[-1].replace('.fits', '')
    
    pv = PVAnalysis()
    pv.get_edgeridge(pvfits=pvfits, incl=incl, vsys=vsys, dist=dist,
                     sigma=sigma, cutoff=cutoff, quadrant=quadrant,
                     Mlim=Mlim, xlim=xlim, vlim=vlim,
                     use_velocity=use_velocity,
                     use_position=use_position,
                     minrelerr=minrelerr, minabserr=minabserr,
                     ridge=ridge, interp_ridge=interp_ridge)
    if write_point: pv.write_edgeridge(filehead)
    pv.fit_edgeridge(include_vsys=include_vsys, include_dp=include_dp,
                     include_pin=include_pin, show_corner=show_corner,
                     filehead=filehead)
    pv.output_fitresult()
    pv.plot_edgeridge(xlim=xlim_plot, vlim=vlim_plot,
                      show_pv=show_pv, filehead=filehead)
