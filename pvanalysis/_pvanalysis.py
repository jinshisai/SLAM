# -*- coding: utf-8 -*-

'''
Program to perform Gaussian fitting to a PV diagram.
Made and developed by Yusuke ASO & Jinshi Sai.

E-mail: jn.insa.sai@gmail.com

Latest update: 3/2/2022
'''


# modules
import copy
import numpy as np
import matplotlib.pyplot as plt
from astropy import constants, units

from .fitfuncs import edge, ridge_mean, gaussfit, gauss1d
from pvanalysis.pvfits import Impvfits
from pvanalysis.pvplot import PVPlot
from pvanalysis.analysis_tools import doublepower_r, doublepower_v, doublepower_v_error, doublepower_r_error
from utils import emcee_corner


# Constants (in cgs)
Ggrav  = constants.G.cgs.value     # Gravitational constant
Msun   = constants.M_sun.cgs.value # Solar mass (g)
au     = units.au.to('cm')         # au (cm)
clight = constants.c.cgs.value     # light speed (cm s^-1)



class PVAnalysis():
    """Perform fitting to a rotational profile using a PV diagram.

    Args:
        infile (str): Input fits file.
        rms (float): RMS noise level of the pv diagram.
        vsys (float): Systemic velocity of the object (km s^-1).
        dist (float): Distance to the object (pc).
    """
    def __init__(self, infile, rms, vsys, dist, pa=None):
        """Initialize.

        Args:
            infile (str): An input fits file.
            rms (float): rms noise level of the pv diagram.
            vsys (float): Systemic velocity of the object (km s^-1).
            dist (float): Distance to the object (pc).
            pa (float, optional): Position angle of the pv cut.
               Will be used to calculate the spatial resolution along the pv
               cut if it is given. Defaults to None.
        """
        # read fits file
        self.fitsdata = Impvfits(infile, pa=pa)
        # parameters required for analysis
        self.rms  = rms
        self.vsys = vsys
        self.dist = dist
        # initialize results
        self.results = {'ridge': {'vcut': None, 'xcut': None},
                        'edge': {'vcut': None, 'xcut': None}}
        self.__sorted = False

    def get_edgeridge(self, outname, thr=5.,
        incl=90., quadrant=None, mode='mean',
        pixrng_vcut=None, pixrng_xcut=None,
        Mlim=[0, 1e10], xlim=[-1e10, 0, 0, 1e10], vlim=[-1e10, 0, 0, 1e10],
        use_velocity=True, use_position=True,
        interp_ridge=False, minrelerr=0.01, minabserr=0.1):
        """Get the edge/ridge at positions/velocities.

        Args:
            outname (str): Output file name.
            thr (float, optional): Threshold above which data is used
               to get edge/ridge. The threshold will be used as thr x rms.
               Defaults to 5.
            incl (float, optional): Inclination angle of the object.
               Defaults to 90, which means no correction for estimate
               of the protostellar mass.
            quadrant (str, optional): Quadrant of the PV diagram where you
               want to get edge/ridge. Defaults to None.
            mode (str, optional): Method to derive ridge. Must be 'mean' or
               'gauss'. Defaults to 'mean'.
            pixrng_vcut (int, optional): Pixel range within which gaussian
               fit is performed around the peak intensity along the v-axis.
               Only used when mode='gauss'. Defaults to None.
            pixrng_xcut (int, optional): Pixel range within which gaussian
               fit is performed around the peak intensity along the x-axis.
               Only used when mode='gauss'. Defaults to None.
            Mlim (list, optional): Reliable mass range. Data points that do
               not come within this range is removed. Defaults to [0, 1e10].
            xlim (list, optional): Range of offset where edge/ridge is
               derived. Defaults to [-1e10, 0, 0, 1e10].
            vlim (list, optional): Range of velocity where edge/ridge is
               derived. Defaults to [-1e10, 0, 0, 1e10].
            use_velocity (bool, optional): Derive representative velocities
               as a function of position. Defaults to True.
            use_position (bool, optional): Derive representatieve positions
               as a function of velocity. Defaults to True.
            interp_ridge (bool, optional): _description_. Defaults to False.
        """
        # remember output name
        self.outname = outname
        # data
        data = self.fitsdata.data
        nxh  = self.fitsdata.nx // 2
        nvh  = self.fitsdata.nv // 2
        if not (self.fitsdata.naxis in [2, 3]):
            print('ERROR\tget_edgeridge: n_axis must be 2 or 3.')
            return
        elif self.fitsdata.naxis == 3:
            data = np.squeeze(data) # Remove stokes I
        # check quadrant
        quadcheck = lambda a: (np.sum(a[:nvh, :nxh])
                               + np.sum(a[nvh:, nxh:])
                               - np.sum(a[:nvh, nxh:])
                               - np.sum(a[nvh:, :nxh]))
        q = np.sign(quadcheck(data))
        if quadrant is None:
            self.quadrant = '13' if q > 0 else '24'
        else:
            self.quadrant = quadrant
        if self.quadrant == '13':
            self.xsign = 1.
        elif self.quadrant == '24':
            self.xsign = -1.
        else:
            print('ERROR\tget_edgeridge: quadrant must be 13 or 24.')
            return
        if self.xsign != q:
            print('WARNING\tget_edgeridge: '
                  + f'quadrant={q:.0f} seems opposite.')
        # store fitting conditions
        self.xlim = xlim
        self.vlim = vlim
        self.Mlim = Mlim
        self.__unit = 1e10 * self.dist * au \
                      / Ggrav / Msun / np.sin(np.radians(incl))**2
        self.__use_position = use_position
        self.__use_velocity = use_velocity

        # Get rigde/edge
        if use_position:
            self.get_edgeridge_xcut(outname, thr=thr, incl=incl,
                            xlim=xlim, vlim=vlim, Mlim=Mlim,
                            mode=mode, pixrng=pixrng_xcut,
                            interp_ridge=interp_ridge)
        if use_velocity:
            self.get_edgeridge_vcut(outname, thr=thr, incl=incl,
                            xlim=xlim, vlim=vlim, Mlim=Mlim,
                            mode=mode, pixrng=pixrng_vcut,
                            interp_ridge=interp_ridge)

        # sort results
        self.sort_fitresults(minrelerr=minrelerr, minabserr=minabserr)
        # plot
        self.plotresults_pvdiagram()
        self.plotresults_rvplane()

    def sort_fitresults(self, minrelerr=0.01, minabserr=0.1):
        """Sort fitting results.

        Args:
            minrelerr, minabserr (float): Parameters to clip too small errors.
        """
        # results are sorted?
        self.__sorted = True
        # to store
        self.results_sorted = {'ridge': {'red': None, 'blue': None},
                               'edge': {'red': None, 'blue': None}}
        self.results_filtered = {'ridge': None, 'edge': None}

        # error clip
        def clipped_error(err, val, mode):
            res    = self.res_off if mode == 'x' else self.delv
            minabs = [minabserr * res] * len(err)
            return np.max([err, minrelerr * np.abs(val), minabs], axis=0)

        # sort results
        for i in ['ridge', 'edge']:
            # tentative space
            self.__store = {'xcut': {'red': None, 'blue': None},
                            'vcut': {'red': None, 'blue': None}}
            for j in ['xcut', 'vcut']:
                if self.results[i][j] is None:
                    continue
                # x, v, err_x, err_v
                results     = copy.deepcopy(self.results[i][j])
                results[1] -= self.vsys
                # clip too small error
                if j == 'xcut':
                    # clip x error
                    results[2] = clipped_error(results[2], results[0], mode='x')
                else:
                    # clip v error
                    results[3] = clipped_error(results[3], results[1], mode='v')
                # relative velocity to separate red/blue comp.
                vrel = results[1]
                xrel = results[0]
                # separate red/blue components
                rel = vrel if j == 'xcut' else xrel * self.xsign
                redsign = int(self.xsign) if j == 'vcut' else 1
                res_red  = [k[rel > 0][::redsign] for k in results]
                res_blue = [k[rel < 0][::-redsign] for k in results]
                ## nan after turn over
                jj = 0 if j == 'xcut' else 1
                for a in [res_red[jj], res_blue[jj]]:
                    if np.any(~np.isnan(a)):
                        a[:np.nanargmax(np.abs(a))] = np.nan
                # remove points outside Mlim
                if len(self.Mlim) == 2:
                    for a in [res_red, res_blue]:
                        mass_est = kepler_mass(a[0], a[1], self.__unit)
                        a[jj][~between(mass_est, self.Mlim)] = np.nan
                self.__store[j]['red']  = res_red
                self.__store[j]['blue'] = res_blue
            # remove low-velocity positions and inner velocities when using both positions and velocities
            if ((self.results[i]['xcut'] is not None) 
                & (self.results[i]['vcut'] is not None)):
                for j in ['red', 'blue']:
                    x1, v0, _, _ = self.__store['xcut'][j]
                    x0, v1, _, _ = self.__store['vcut'][j]
                    xx, xv = np.meshgrid(x1, x0)
                    vx, vv = np.meshgrid(v0, v1)
                    xvd = np.hypot((xx-xv) / self.res_off, (vx-vv) / self.delv)
                    if np.any(~np.isnan(xvd)):
                        iv, ix = np.unravel_index(np.nanargmin(xvd), np.shape(xx))
                        x1[:ix] = np.nan
                        v1[:iv] = np.nan
            # combine xcut and vcut
            res_f = {'xcut':{'red':np.array([[], [], [], []]),
                             'blue':np.array([[], [], [], []])},
                     'vcut':{'red':np.array([[], [], [], []]),
                             'blue':np.array([[], [], [], []])}}
            for j in ['red', 'blue']:
                if ((self.results[i]['xcut'] is not None) 
                    & (self.results[i]['vcut'] is not None)):
                    # remove nan
                    for cut, jj in zip(['xcut', 'vcut'], [0, 1]):
                        ref = self.__store[cut][j]
                        store = [k[~np.isnan(ref[jj])] for k in ref]
                        self.__store[cut][j] = store
                        res_f[cut][j] = store
                    # combine xcut/vcut
                    res_comb = np.array(
                        [np.append(self.__store['xcut'][j][k],
                                   self.__store['vcut'][j][k])
                         for k in range(4)])
                elif not ((self.results[i]['xcut'] is None)
                          and (self.results[i]['vcut'] is None)):
                    # only remove nan
                    if self.results[i]['xcut'] is not None:
                        cut, jj, ref = 'xcut', 0, self.__store['xcut'][j]
                    else:
                        cut, jj, ref = 'vcut', 1, self.__store['vcut'][j]
                    res_comb = [k[~np.isnan(ref[jj])] for k in ref]
                    res_f[cut][j] = [k[~np.isnan(ref[jj])] for k in ref]
                else:
                    print('ERROR\tsort_fitresults: '
                          + 'No fitting results are found.')
                    return

                # arcsec --> au
                res_comb[0] = res_comb[0]*self.dist
                res_comb[2] = res_comb[2]*self.dist
                for cut in ['xcut', 'vcut']:
                    res_f[cut][j][0] = res_f[cut][j][0]*self.dist
                    res_f[cut][j][2] = res_f[cut][j][2]*self.dist
                ## sort by x
                i_order  = np.argsort(np.abs(res_comb[0]))
                res_comb = np.array([np.abs(k[i_order]) for k in res_comb])
                # save
                self.results_sorted[i][j] = res_comb
            self.results_filtered[i] = res_f
        del self.__store

    def get_edgeridge_vcut(self, outname, thr=5., incl=90., xlim=[-1e10, 0, 0, 1e10],
                   vlim=[-1e10, 0, 0, 1e10], Mlim=[0, 1e10], mode='gauss',
                   pixrng=None, multipeaks=False, i_peak=0, prominence=1.5,
                   inverse=False, interp_ridge=False):
        """Get edge/ridge along the velocity axis, i.e., determine
           representative velocity at each offset.

        Args:
            outname (str): Output file name.
            thr (float): Threshold for the fitting. Spectrum with the maximum
               intensity above thr x rms will be used for the fitting.
               xlim, vlim: x and v ranges for the fitting. Must be given
               as a list, [outlimit1, inlimit1, inlimit2, outlimit2].
            incl (float): Inclination angle of the object. Defaults to 90,
               which means no correction for estimate of the protostellar
               mass.
            Mlim (list): Reliable mass range. Data points that do
               not come within this range is removed. Defaults to [0, 1e10].
            xlim (list): Range of offset where edge/ridge is
               derived. Defaults to [-1e10, 0, 0, 1e10].
            vlim (list): Range of velocity where edge/ridge is
               derived. Defaults to [-1e10, 0, 0, 1e10].
            mode (str): Method to derive ridge. When mode='mean', ridge is
               derived as the intensity weighted mean. When mode='gauss',
               ridge is derived as the mean of the fitted Gaussian function.
               Defaults to 'mean'.
            pixrng (float): Pixel range for the fitting around the maximum
               intensity. Only velocity channels +/- pixrng around the channel
               with the maximum intensity are used for the fitting.
               Only applied when mode='gauss'.
            multipeaks (bool): Find multiple peaks if True. Default False,
               which means only the maximum peak is considered.
               Only applied when mode='gauss'.
            i_peak (int): Only used when multipeaks=True. Used to specify
               around which intensity peak you want to perform the fitting.
            prominence (float): Only used when multipeaks=True. Parameter to
               determine multiple peaks.
            inverse (bool): Inverse axes. May be used to sort the sampling
               between fittings with different xlim or vlim.
            interp_ridge (bool): If True, vaxis is interporated with spline
               interpolation to derive ridge. Deafaults to False.
        """
        # modules
        import math
        from scipy.signal import find_peaks
        from scipy.interpolate import interp1d
        from mpl_toolkits.axes_grid1 import ImageGrid

        rms = self.rms

        print('PV fit along velocity axis.')
        # data
        data = self.fitsdata.data
        if not (self.fitsdata.naxis in [2, 3]):
            print('ERROR\tpvfit_vcut: n_axis must be 2 or 3.')
            return
        if self.fitsdata.naxis == 3:
            data = np.squeeze(data) # Select stokes I
        # axes
        xaxis = self.fitsdata.xaxis
        vaxis = self.fitsdata.vaxis
        # resolution
        if self.fitsdata.res_off:
            res_off = self.fitsdata.res_off
        else:
            bmaj, bmin, bpa = self.fitsdata.beam
            res_off = bmaj # in arcsec
        self.res_off = res_off
        self.delv = self.fitsdata.delv
        # harf of beamsize [pix]
        hob = int(np.round((res_off*0.5/self.fitsdata.delx)))

        ### calculate intensity-weighted mean positions
        # cutting at an velocity and determining the position

        # x & v ranges used for calculation for fitting
        xaxis_fit = xaxis
        vaxis_fit = vaxis
        data_fit  = data
        if len(xlim) != 4:
            print('Warning\tpvfit_vcut: '
                  + 'Size of xlim is not correct. '
                  + 'xlim must be given as [x0, x1, x2, x3], '
                  + 'meaning (x0, x1) and (x2, x3) are used. '
                  + 'Given xlim is ignored.')
            xlim = (-1e10, 0, 0, 1e10)
        else:
            b = between(xaxis, (xlim[0], xlim[3]))
            xaxis_fit = xaxis[b]
            data_fit  = np.array([d[b] for d in data])
            xmin, xmax = np.min(xaxis_fit), np.max(xaxis_fit)
            print(f'x range: {xmin:.2f} -- {xmax:.2f} arcsec')
        vmin, vmax = np.min(vaxis_fit), np.max(vaxis_fit)
        if len(vlim) != 4:
            print('Warning\tpvfit_vcut: '
                  + 'Size of vlim is not correct. '
                  + 'vlim must be given as [v0, v1, v2, v3], '
                  + 'meaning (v0, v1) and (v2, v3) are used. '
                  + 'Given vlim is ignored.')
            vlim = (-1e10, 0, 0, 1e10)
        else:
            b = between(vaxis, (vlim[0], vlim[3]))
            vaxis_fit = vaxis[b]
            data_fit  = np.array([d[b] for d in data_fit.T]).T
            vmin, vmax = np.min(vaxis_fit), np.max(vaxis_fit)
            print(f'v range: {vmin:.2f} -- {vmax:.2f} km/s')
        # to achieve the same sampling on two sides
        if inverse:
            xaxis_fit = xaxis_fit[::-1]
            data_fit  = data_fit[:, ::-1]
        # Nyquist sampling
        xaxis_fit = xaxis_fit[::hob]
        data_fit  = data_fit[:, ::hob]
        nloop     = len(xaxis_fit)
        ncol = int(math.ceil(np.sqrt(nloop)))
        nrow = int(math.ceil(nloop / ncol))
        # figure for check result
        fig  = plt.figure(figsize=(11.69, 8.27))
        grid = ImageGrid(fig, rect=111, nrows_ncols=(nrow, ncol),
            axes_pad=0,share_all=True, aspect=False, label_mode='L')
        dlim  = [np.nanmin(data_fit), np.nanmax(data_fit)]
        # x & y label
        grid[(nrow*ncol - ncol)].set_xlabel(r'Velocity (km s$^{-1}$)')
        grid[(nrow*ncol - ncol)].set_ylabel('Intensity')
        # list to save final results
        res_ridge = np.empty((nloop, 4))
        res_edge  = np.empty((nloop, 4))

        vi_interp = np.linspace(vaxis_fit[0], vaxis_fit[-1],
                                (len(vaxis_fit) - 1) * 10 + 1)
        v_i_raw = vaxis_fit.copy()  # for plot
        # loop for x
        for i in range(nloop):
            # ith data
            x_i = xaxis_fit[i]
            d_i = data_fit[:, i].copy()
            if np.all(np.isnan(d_i)) or (xlim[1] < x_i < xlim[2]):
                res_ridge[i, :] = [x_i, np.nan, 0, np.nan]
                res_edge[i, :] = [x_i, np.nan, 0, np.nan]
                continue
            # plot results
            ax  = grid[i]
            # get ridge value
            # interpolate
            v_i = vi_interp.copy() if interp_ridge else v_i_raw
            di_interp = interp1d(v_i_raw, d_i, kind='cubic')(vi_interp)
            # if interpolate
            d_i_raw = d_i.copy() # for plot
            d_i = di_interp.copy() if interp_ridge else d_i
            # ridge
            if mode == 'mean':
                c = (d_i >= thr * rms)
                v_i, d_i = v_i[c], d_i[c]
                mv, mv_err = ridge_mean(v_i, d_i, rms)
                # plot
                if ~np.isnan(mv):
                    ax.vlines(mv, 0., dlim[-1], lw=1.5,
                              color='r', ls='--', alpha=0.8)
            elif mode == 'gauss':
                # get peak indices
                peaks, _ = find_peaks(d_i, height=thr * rms,
                                      prominence=prominence * rms)
                # determining used data range
                if pixrng:
                    # error check
                    if type(pixrng) != int:
                        print('ERROR\tpvfit_vcut: '
                              + 'pixrng must be integer.')
                        return
                    # get peak index
                    pidx = peaks[i_peak] if multipeaks else np.argmax(d_i)
                    if ((pidx < pixrng)
                        or (pidx > len(d_i) - pixrng - 1)):
                        # peak position is too close to edge
                        mv, mv_err = np.nan, np.nan
                    else:
                        # use pixels only around intensity peak
                        v0, v1 = pidx - pixrng, pidx + pixrng + 1
                        d_i, v_i  = d_i[v0:v1], v_i[v0:v1]
                        #nd_i = len(d_i)

                if np.nanmax(d_i) >= thr * rms:
                    popt, perr = gaussfit(v_i, d_i, rms)
                else:
                    popt, perr = np.full(3, np.nan), np.full(3, np.nan)
                mv, mv_err = popt[1], perr[1]
                # Plot result
                if ~np.isnan(mv):
                    v_model = np.linspace(vmin, vmax, 256) # offset axis for plot
                    g_model = gauss1d(v_model, *popt)
                    ax.step(v_i, d_i, linewidth=1.5, color='r',
                            where='mid')
                    ax.plot(v_model, g_model, lw=1.5, color='r',
                            ls='-', alpha=0.8)
                    ax.vlines(mv, 0., dlim[-1], lw=1.5, color='r',
                              ls='--', alpha=0.8)
            else:
                print('ERROR\tpvfit_vcut: mode must be mean or gauss.')
                return
            if not (vlim[0] < mv < vlim[1] or vlim[2] < mv < vlim[3]):
                mv, mv_err = np.nan, np.nan
            if interp_ridge: mv_err *= np.sqrt(10.)
            # output ridge results
            res_ridge[i, :] = [x_i, mv, 0, mv_err]

            # get edge values
            # fitting axis
            v_i = vaxis_fit.copy()
            d_i = data_fit[:,i].copy()
            # interpolation
            # resample with a 10 times finer sampling rate
            vi_interp  = np.linspace(v_i[0], v_i[-1],
                                     (len(v_i)-1)*10 + 1)
            di_interp  = interp1d(v_i, d_i, kind='cubic')(vi_interp)
            # flag by mass
            mass_est = kepler_mass(x_i, vi_interp-self.vsys, self.__unit)
            goodflag = between(mass_est, Mlim) if len(Mlim) == 2 else None
            edgesign = v_i[np.nanargmax(d_i)] - self.vsys
            mv, mv_err = edge(vi_interp, di_interp, rms, rms*thr,
                              goodflag=goodflag, edgesign=edgesign)
            if not (vlim[0] < mv < vlim[1] or vlim[2] < mv < vlim[3]):
                mv, mv_err = np.nan, np.nan
            res_edge[i, :] = [x_i, mv, 0, mv_err]
            # plot
            if ~np.isnan(mv):
                ax.vlines(mv, 0., dlim[-1], lw=1.5, color='b',
                          ls='--', alpha=0.8)
            # observed data
            ax.step(v_i_raw, d_i_raw, linewidth=1.,
                    color='k', where='mid')
            # offset label
            ax.text(0.9, 0.9, f'{x_i:03.2f}', horizontalalignment='right',
                verticalalignment='top',transform=ax.transAxes)
            ax.tick_params(which='both', direction='in',bottom=True,
                           top=True, left=True, right=True, pad=9)
        # Store the result array in the shape of (4, len(x)).
        self.results['ridge']['vcut'] = res_ridge.T
        self.results['edge']['vcut']  = res_edge.T
        #plt.show()
        fig.savefig(outname + "_pvfit_vcut.pdf", transparent=True)
        plt.close()


    def get_edgeridge_xcut(self, outname, thr=5., incl=90., xlim=[-1e10, 0, 0, 1e10],
                   vlim=[-1e10, 0, 0, 1e10], Mlim=[0, 1e10], mode='mean',
                   pixrng=None, multipeaks=False, i_peak=0,
                   prominence=1.5, interp_ridge=False):
        """Get edge/ridge along x-axis, i.e., determine representative
           position at each velocity.

        Args:
            outname (str): Output file name.
            thr (float): Threshold for the fitting. Spectrum with the maximum
               intensity above thr x rms will be used for the fitting.
               xlim, vlim: x and v ranges for the fitting. Must be given
               as a list, [outlimit1, inlimit1, inlimit2, outlimit2].
            incl (float): Inclination angle of the object. Defaults to 90,
               which means no correction for estimate of the protostellar
               mass.
            Mlim (list): Reliable mass range. Data points that do
               not come within this range is removed. Defaults to [0, 1e10].
            xlim (list): Range of offset where edge/ridge is
               derived. Defaults to [-1e10, 0, 0, 1e10].
            vlim (list): Range of velocity where edge/ridge is
               derived. Defaults to [-1e10, 0, 0, 1e10].
            mode (str): Method to derive ridge. When mode='mean', ridge is
               derived as the intensity weighted mean. When mode='gauss',
               ridge is derived as the mean of the fitted Gaussian function.
               Defaults to 'mean'.
            pixrng (float): Pixel range for the fitting around the maximum
               intensity. Only pixels +/- pixrng around the (re-samped) pixel
               with the maximum intensity are used for the fitting.
               Only applied when mode='gauss'.
            multipeaks (bool): Find multiple peaks if True. Default False,
               which means only the maximum peak is considered.
               Only applied when mode='gauss'.
            i_peak (int): Only used when multipeaks=True. Used to specify
               around which intensity peak you want to perform the fitting.
            prominence (float): Only used when multipeaks=True. Parameter to
               determine multiple peaks.
            inverse (bool): Inverse axes. May be used to sort the sampling
               between fittings with different xlim or vlim.
            interp_ridge (bool): If True, xaxis is not sampled with the
            Nyquist rate to derive ridge. Deafaults to False.
        """
        # modules
        import math
        from scipy import optimize
        from scipy.signal import find_peaks
        from scipy.interpolate import interp1d
        from mpl_toolkits.axes_grid1 import ImageGrid

        rms = self.rms

        print('pvfit along position axis.')

        # data
        data = self.fitsdata.data
        if not (self.fitsdata.naxis in [2, 3]):
            print('Error\tpvfit_vcut: n_axis must be 2 or 3.')
            return
        elif self.fitsdata.naxis == 3:
            data = np.squeeze(data) # Remove stokes I
        # axes
        xaxis = self.fitsdata.xaxis
        vaxis = self.fitsdata.vaxis
        # resolution
        if self.fitsdata.res_off:
            res_off = self.fitsdata.res_off
        else:
            bmaj, bmin, bpa = self.fitsdata.beam
            res_off = bmaj  # in arcsec
        self.res_off = res_off
        self.delv = self.fitsdata.delv
        # harf of beamsize [pix]
        hob  = int(np.round((res_off*0.5/self.fitsdata.delx)))
        # x & v ranges used for calculation for fitting
        xaxis_fit = xaxis
        vaxis_fit = vaxis
        data_fit  = data
        xmin, xmax = np.min(xaxis_fit), np.max(xaxis_fit)
        if len(xlim) != 4:
            print('Warning\tpvfit_vcut: '
                  + 'Size of xlim is not correct. '
                  + 'xlim must be given as [x0, x1, x2, x3], '
                  + 'meaning (x0, x1) and (x2, x3) are used. '
                  + 'Given xlim is ignored.')
            xlim = (-1e10, 0, 0, 1e10)
        else:  # between can treat tlim=[] now.
            b = between(xaxis, (xlim[0], xlim[3]))
            xaxis_fit = xaxis[b]
            data_fit  = np.array([d[b] for d in data])
            xmin, xmax = np.min(xaxis_fit), np.max(xaxis_fit)
            print(f'x range: {xmin:.2f} -- {xmax:.2f} arcsec')

        if len(vlim) != 4:
            print('Warning\tpvfit_vcut: '
                  + 'Size of vlim is not correct. '
                  + 'vlim must be given as [v0, v1, v2, v3], '
                  + 'meaning (v0, v1) and (v2, v3) are used. '
                  + 'Given vlim is ignored.')
            vlim = (-1e10, 0, 0, 1e10)
        else:
            b = between(vaxis, (vlim[0], vlim[3]))
            vaxis_fit = vaxis[b]
            data_fit  = np.array([d[b] for d in data_fit.T]).T
            vmin, vmax = np.min(vaxis_fit), np.max(vaxis_fit)
            print(f'v range: {vmin:.2f} -- {vmax:.2f} km/s')
        # Nyquist sampling
        xaxis_fit = xaxis.copy() if interp_ridge else xaxis[::hob]
        vaxis_fit = vaxis.copy()
        data_fit  = data.copy() if interp_ridge else data[:,::hob].copy()
        nloop = len(vaxis_fit)
        ncol = int(math.ceil(np.sqrt(nloop)))
        nrow = int(math.ceil(nloop / ncol))
        # figure for check result
        fig  = plt.figure(figsize=(11.69, 8.27))
        grid = ImageGrid(fig, rect=111, nrows_ncols=(nrow,ncol),
            axes_pad=0,share_all=True, aspect=False, label_mode='L')
        #gridi = 0
        # x & y label
        grid[(nrow*ncol-ncol)].set_xlabel('Offset (arcsec)')
        grid[(nrow*ncol-ncol)].set_ylabel('Intensity')
        dlim = [np.nanmin(data_fit), np.nanmax(data_fit)]
        # list to save final results
        res_ridge = np.empty((nloop, 4))
        res_edge  = np.empty((nloop, 4))

        # loop for v
        for i in range(nloop):
            # ith data
            v_i  = vaxis_fit[i]
            d_i  = data_fit[i, :]
            if np.all(np.isnan(d_i)) or (vlim[1] < v_i < vlim[2]):
                res_ridge[i, :] = [np.nan, v_i, np.nan, 0]
                res_edge[i, :] = [np.nan, v_i, np.nan, 0]
                continue
            # plot results
            ax = grid[i]

            # get ridge value
            if mode == 'mean':
                c = (d_i >= thr*rms)
                x_i, d_i = xaxis_fit[c], d_i[c]
                mx, mx_err = ridge_mean(x_i, d_i, rms)
                # plot
                if ~np.isnan(mx):
                    ax.vlines(mx, 0., dlim[-1], lw=1.5, color='r',
                              ls='--', alpha=0.8)
            elif mode == 'gauss':
                # get peak indices
                peaks, _ = find_peaks(d_i, height=thr * rms,
                                      prominence=prominence * rms)
                # determining used data range
                if pixrng:
                    # error check
                    if type(pixrng) != int:
                        print('ERROR\tpvfit_xcut: pixrng must be integer.')
                        return
                    # get peak index
                    pidx = peaks[i_peak] if multipeaks else np.argmax(d_i)
                    if ((pidx < pixrng)
                        or (pidx > len(d_i) - pixrng - 1)):
                        # peak position is too edge
                        mx, mx_err = np.nan, np.nan
                    else:
                        # use pixels only around intensity peak
                        x0, x1 = pidx - pixrng, pidx + pixrng + 1
                        d_i, x_i  = d_i[x0:x1], xaxis_fit[x0:x1]
                        #nd_i = len(d_i)
                else:
                    x_i = xaxis_fit.copy()
                if np.nanmax(d_i) >= thr * rms:
                    popt, perr = gaussfit(x_i, d_i, rms)
                else:
                    popt, perr = np.full(3, np.nan), np.full(3, np.nan)
                mx, mx_err = popt[1], perr[1]
                # Plot result
                if ~np.isnan(mx):
                    x_model = np.linspace(xmin, xmax, 256) # offset axis for plot
                    g_model = gauss1d(x_model, *popt)
                    ax.step(x_i, d_i, linewidth=1.5, color='r',
                            where='mid')
                    ax.plot(x_model, g_model, lw=1.5, color='r',
                            ls='-', alpha=0.8)
                    ax.vlines(mx, 0., dlim[-1], lw=1.5, color='r',
                              ls='--', alpha=0.8)
            else:
                print('ERROR\tpvfit_xcut: mode must be mean or gauss.')
                return
            if not (xlim[0] < mx < xlim[1] or xlim[2] < mx < xlim[3]):
                mx, mx_err = np.nan, np.nan
            if interp_ridge: mx_err *= np.sqrt(hob)
            # output ridge results                
            res_ridge[i, :] = [mx, v_i, mx_err, 0.]

            # get edge values
            # fitting axes
            x_i = xaxis_fit.copy()
            d_i = data_fit[i, :].copy()
            # interpolation
            # resample with a 10 times finer sampling rate
            xi_interp  = np.linspace(x_i[0], x_i[-1],
                                     (len(x_i)-1)*10 + 1)
            di_interp  = interp1d(x_i, d_i, kind='cubic')(xi_interp)
            # flag by mass
            mass_est = kepler_mass(xi_interp, v_i - self.vsys, self.__unit)
            goodflag = between(mass_est, Mlim) if len(Mlim) == 2 else None
            edgesign = x_i[np.nanargmax(d_i)]
            mx, mx_err = edge(xi_interp, di_interp, rms, rms*thr,
                              goodflag=goodflag, edgesign=edgesign)
            if not (xlim[0] < mx < xlim[1] or xlim[2] < mx < xlim[3]):
                mx, mx_err = np.nan, np.nan
            res_edge[i, :] = [mx, v_i, mx_err, 0.]
            # plot
            if ~np.isnan(mx):
                ax.vlines(mx, 0., dlim[-1], lw=2., color='b',
                          ls='--', alpha=0.8)
            # observed data
            ax.step(xaxis_fit, data_fit[i,:], linewidth=1.,
                    color='k', where='mid')
            # offset label
            ax.text(0.9, 0.9, f'{v_i:03.2f}', horizontalalignment='right',
                verticalalignment='top',transform=ax.transAxes)
            ax.tick_params(which='both', direction='in', bottom=True,
                           top=True, left=True, right=True, pad=9)
        # Store the result array in the shape of (4, len(v))
        self.results['ridge']['xcut'] = res_ridge.T
        self.results['edge']['xcut']  = res_edge.T
        #plt.show()
        fig.savefig(outname+"_pvfit_xcut.pdf", transparent=True)
        plt.close()
        # output results as .txt file
        #res_hd  = 'offset[arcsec]\tvelocity[km/s]\toff_err[arcesc]\tvel_err[km/s]'
        #res_all = np.c_[res_x, res_v, err_x, err_v]
        #np.savetxt(outname+'_chi2_pv_xfit.txt', res_all,fmt = '%4.4f', delimiter ='\t', header = res_hd)

    def fit_edgeridge(self, include_vsys: bool = False,
                      include_dp: bool = True,
                      include_pin: bool = False,
                      filehead: str = 'pvanalysis',
                      show_corner: bool = False):
        """Fit the derived edge/ridge positions/velocities with a double power law function by using emcee.


        Args:
            include_vsys : bool
               False means vsys is fixed at 0.
            include_dp : bool
               False means dp is fixed at 0, i.e., single power law function.
            include_pin : bool
               False means pin is fixed at 0.5, i.e., the Keplerian law.
            filehead : str
               The output corner figures have names of "filehead".corner_e.png
               and "filehead".corner_r.png.
            show_corner : bool
               True means the corner figures are shown. These figures are also
               plotted in two png files.

        Returns:
            fitsdata : dict
                {'edge':{'popt':[...], 'perr':[...]}, 'ridge':{...}}
                'popt' is a list of [r_break, v_break, p_in, dp, vsys].
                'perr' is the uncertainties for popt.
        """
        res_org = self.results_filtered
        Ds = [None, None]
        for i, er in enumerate(['edge', 'ridge']):
            xrb  = [res_org[er]['xcut'][rb] for rb in ['red', 'blue']]
            xcut = np.concatenate(xrb, axis=1)
            v0, x1, dx1 = xcut[1], self.xsign * xcut[0], xcut[2]
            vrb  = [res_org[er]['vcut'][rb] for rb in ['red', 'blue']]
            vcut = np.concatenate(vrb, axis=1)
            x0, v1, dv1 = self.xsign * vcut[0], vcut[1], vcut[3]
            Ds[i] = [v0, x1, dx1, x0, v1, dv1]
        Es, Rs = Ds
        self.__Es = Es
        self.__Rs = Rs
        self.include_dp = include_dp

        labels = np.array(['Rb', 'Vb', 'p_in', 'dp', 'Vsys'])
        include = [True, include_dp, include_pin, include_dp, include_vsys]
        labels = labels[include]
        popt_e, popt_r = [np.empty(5), np.empty(5)], [np.empty(5), np.empty(5)]
        minabs = lambda a, i, j: np.min(np.abs(np.r_[a[i], a[j]]))
        maxabs = lambda a, i, j: np.max(np.abs(np.r_[a[i], a[j]]))
        for args, ext, res in zip([Es, Rs], ['_e', '_r'], [popt_e, popt_r]):
            plim = np.array([
                   [minabs(args, 1, 3), minabs(args, 0, 4), 0.01, 0, -1],
                   [maxabs(args, 1, 3), maxabs(args, 0, 4), 10.0, 10, 1]])
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
        print(f'corner plots in {filehead}.corner_e.png '
              + f'and {filehead}.corner_r.png')
        self.popt = {'edge':popt_e, 'ridge':popt_r}
        self.model = doublepower_v
        result = {'edge':{'popt':popt_e[0], 'perr':popt_e[1]},
                  'ridge':{'popt':popt_r[0], 'perr':popt_r[1]}}
        return result

    def write_edgeridge(self, filehead='pvanalysis'):
        """Write the edge/ridge positions/velocities to a text file.

        Args:
            filehead (str): The output text file has a name of
               "filehead".edge.dat. and "filehead".ridge.dat The file
               consists of four columns of x (au), dx (au), v (km/s),
               and dv (km/s).
        """
        res_org = self.results_filtered
        for i, er in enumerate(['edge', 'ridge']):
            xrb = [res_org[er]['xcut'][rb] for rb in ['red', 'blue']]
            xcut = np.concatenate(xrb, axis=1)
            v0, x1, dv0, dx1 = xcut[1], self.xsign * xcut[0], xcut[3], xcut[2]
            vrb = [res_org[er]['vcut'][rb] for rb in ['red', 'blue']]
            vcut = np.concatenate(vrb, axis=1)
            x0, v1, dx0, dv1 = self.xsign * vcut[0], vcut[1], vcut[2], vcut[3]
            np.savetxt(filehead + '.' + er + '.dat',
                       np.c_[np.r_[x1, x0], np.r_[dx1, dx0],
                             np.r_[v0, v1], np.r_[dv0, dv1]],
                       header='x (au), dx (au), v (km/s), dv (km/s)')
        print(f'- Wrote to {filehead}.edge.dat and {filehead}.ridge.dat.')


    def get_range(self):
        """Calculate the ranges of the edge/ridge positions (radii) and velocities.

        Args:
            No parameter.

        Returns:
            result : dict
            {'edge':{'rlim':[...], 'vlim':[...]}, 'ridge':{...}}
            'rlim' is a list of [r_minimum, r_maximum].
            'vlim' is a list of [v_minimum, v_maximum].
        """
        def inout(v0, x1, dx1, x0, v1, dv1, popt):
            xall, vall = np.abs(np.r_[x0, x1]), np.abs(np.r_[v0, v1])
            rin, rout = np.min(xall), np.max(xall)
            vin, vout = np.max(vall), np.min(vall)
            if self.__use_position:
                rin  = doublepower_r(vin, *popt)
            else:
                vin  = doublepower_v(rin, *popt)
            if self.__use_velocity:
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
        """Output the fitting result in the terminal.

        Args:
            No parameter.

        Returns:
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
            M_in = kepler_mass(rin, vin, self.__unit/self.dist)
            M_b  = kepler_mass(rb, vb, self.__unit/self.dist)
            M_out = kepler_mass(rout, vout, self.__unit/self.dist)
            if self.__use_position:
                drin = doublepower_r_error(vin, *params)
                dM_in = M_in * drin / rin
            else:
                dvin = doublepower_v_error(rin, *params)
                dM_in = 2. * M_in * dvin / vin
            if self.__use_velocity:
                dvout = doublepower_v_error(rout, *params)
                dM_out = 2. * M_out * dvout / vout
            else:
                drout = doublepower_r_error(vout, *params)
                dM_out = M_out * drout / rout
            dM_b = kepler_mass_error(rb, vb, drb, dvb, self.__unit/self.dist)
            print(f'M_in  = {M_in:.3f} +/- {dM_in:.3f} Msun')
            print(f'M_out = {M_out:.3f} +/- {dM_out:.3f} Msun')
            print(f'M_b   = {M_b:.3f} +/- {dM_b:.3f} Msun')

    def plot_fitresult(self, vlim: list = [0, 1e10],
                       xlim: list = [0, 1e10],
                       clevels: list = [3, 6],
                       outname: str = 'pvanalysis',
                       show: bool = True):
        for loglog, ext in zip([False, True], ['linear', 'log']):
            pp = PVPlot(restfrq=self.fitsdata.restfreq,
                        beam=self.fitsdata.beam, pa=self.fitsdata.pa,
                        vsys=self.vsys, dist=self.dist,
                        d=self.fitsdata.data,
                        v=self.fitsdata.vaxis, x=self.fitsdata.xaxis,
                        loglog=loglog, vlim=vlim, xlim=xlim)
            self.plot_edgeridge(ax=pp.ax, loglog=loglog)
            self.plot_model(ax=pp.ax, loglog=loglog)
            pp.add_color()
            pp.add_contour(rms=self.rms, levels=clevels)
            pp.set_axis()
            pp.savefig(figname=outname + '.' + ext + '.png', show=show)


    def plot_edgeridge(self, ax=None, loglog: bool = False,
                       flipaxis: bool = False) -> None:
        if ax is None:
            print('Please input ax.')
            return -1
        fmt = {'ridge': 'o', 'edge': 'v'}
        color = {'xcut': {'red': 'red', 'blue': 'blue'},
                 'vcut': {'red': 'pink', 'blue': 'cyan'}}
        for re in ['ridge', 'edge']:
         for xv in ['xcut', 'vcut']:
          for rb in ['red', 'blue']:
            x, v, dx, dv = self.results_filtered[re][xv][rb]
            if xv == 'xcut': dv = 0
            if xv == 'vcut': dx = 0
            if loglog: x, v = np.abs(x), np.abs(v)
            if flipaxis: x, v, dx, dv = v, x, dv, dx
            ax.errorbar(x, v, xerr=dx, yerr=dv, fmt=fmt[re],
                        color=color[xv][rb], ms=5)

    def plot_model(self, ax=None, loglog: bool = False,
                   edge_model = None, edge_popt: list = [],
                   ridge_model = None, ridge_popt: list = [],
                   flipaxis: bool = False) -> None:
        if ax is None:
            print('Please input ax.')
            return -1
        if edge_model is None: edge_model = self.model
        if ridge_model is None: ridge_model = self.model
        if edge_popt  == []: edge_popt = self.popt['edge'][0]
        if ridge_popt == []: ridge_popt = self.popt['ridge'][0]
        rmodel = lambda x: ridge_model(x, *ridge_popt)
        emodel = lambda x: edge_model(x, *edge_popt)
        model = {'ridge': rmodel, 'edge': emodel}
        ls = {'ridge': '-', 'edge': '--'}
        for re in ['ridge', 'edge']:
            xmin, xmax = self.rvlim[re][0]
            if loglog:
                s, x = 1, np.geomspace(xmin, xmax, 100)
            else:
                s, x = self.xsign, np.linspace(-xmax, xmax, 100)
                x[(-xmin < x) * (x < xmin)] = None
            x, y = x, s * model[re](x)
            if flipaxis: x, y = y, x
            ax.plot(x, y, ls=ls[re], lw=2, color='gray', zorder=3)


    # Plot results
    def plotresults_pvdiagram(self, outname=None, marker='o', colors=['r'], alpha=1.,
        ax=None,outformat='pdf',pvcolor=True,cmap='Greys',
        vmin=None,vmax=None, contour=True,clevels=None,pvccolor='k',
        vrel=False,logscale=False,x_offset=False,ratio=1.2, prop_vkep=None,fontsize=14,
        lw=1,clip=None,plot_res=True,inmode='fits',xranges=[], yranges=[],
        ln_hor=True, ln_var=True, pvalpha=None):
        """Plot fitting results on a PV diagram for quick check.


        Args:
            outname (str): Output file name.
            outformat (str): Format of the output file. Deafult pdf.
            ax (mpl axis): Axis of python matplotlib.
            pvcolor (bool): Plot PV diagram in color scale? Default True.
            cmap (str): Color map.
            vmin, vmax (float): Minimum and maximum values for color scale.
            contour (bool): Plot a PV diagram with contour? Default True.
            clevels (list or numpy array): Contour levels. Must be given
               in absolute intensity.
            pvccolor (str): Contour color.
            vrel (bool): Vaxis will be in the relative velocity with respect
               to vsys if True. Defaults to False.
            x_offset (bool): xaxis will be offset axis if True. Default False,
               which means xaxis is velocity axis.
            ratio (float): Aspect ration of the plot.
            lw (float): Line width of the contour.
            xranges, yranges (list): x and y ranges for plot. Given as a list,
               [minimum, maximum].
            marker (str): Marker for the data points.
            colors (str): Colors for the data points. Given as a list.
               If PVFit object has multiple results, more than one colors
               can be given.
            alpha (float): Transparency alpha for the plot.

        Return:
            matplotlib ax.
        """
        # output name
        if outname:
            pass
        else:
            outname = self.outname + '_pvfit_pvdiagram'

        def plotpoints(ax, x, v, xerr, verr, color, x_offset=False):
            if x_offset:
                ax.errorbar(x, v, xerr=xerr, yerr=verr, color=color,
                    capsize=2., capthick=2., ls='', marker='o')
            else:
                ax.errorbar(v, x, xerr=verr, yerr=xerr, color=color,
                    capsize=2., capthick=2., ls='', marker='o')
            return ax

        # draw pv diagram
        ax = self.fitsdata.draw_pvdiagram(outname, data=None, header=None, ax=ax,
            outformat=outformat, color=pvcolor, cmap=cmap,
            vmin=vmin, vmax=vmax, vsys=self.vsys, contour=contour, clevels=clevels,
            ccolor=pvccolor, vrel=vrel, logscale=logscale, x_offset=x_offset,
            ratio=ratio, prop_vkep=prop_vkep, fontsize=fontsize, lw=lw, clip=clip,
            plot_res=plot_res, inmode=inmode, xranges=xranges, yranges=yranges,
            ln_hor=ln_hor, ln_var=ln_var, alpha=pvalpha)


        # plot fitting results
        if self.__sorted:
            results = copy.deepcopy(self.results_sorted)

            # get them back into arcsec and LSR velocity
            for i in results:
                results[i]['red'][0]  *= self.xsign/self.dist
                results[i]['blue'][0] *= -self.xsign/self.dist
                results[i]['red'][2]  *= self.xsign/self.dist
                results[i]['blue'][2] *= -self.xsign/self.dist

                results[i]['red'][1]  = self.vsys + results[i]['red'][1]
                results[i]['blue'][1] = self.vsys - results[i]['blue'][1]

                colors = {'ridge': {'red': 'red', 'blue':'blue'},
                'edge': {'red': 'pink', 'blue': 'skyblue'}}

                for j in results[i]:
                    ax = plotpoints(ax, *results[i][j], colors[i][j], x_offset)
        else:
            results = self.results

            colors = {'ridge':'r', 'edge':'b'}
            for i in results:
                #color = colors[i]
                if self.results[i]['vcut'] is not None:
                    ax = plotpoints(ax, *self.results[i]['vcut'], colors[i], x_offset)

                if self.results[i]['xcut'] is not None:
                    ax = plotpoints(ax, *self.results[i]['vcut'], colors[i], x_offset)

        plt.savefig(outname+'.'+outformat, transparent=True)
        plt.close()
        return ax


    def plotresults_rvplane(self, outname=None, outformat='pdf', ax=None,
        xlog=True, ylog=True, au=False, marker='o',
        capsize=2, capthick=2., colors=None, xlim=[], ylim=[], fontsize=14):
        """Plot edge/ridge in a r-v plane for quick check.

        Args:
            outname (str): Output name.
            outformat (str): Output file format.
            ax (mpl axis): axix object of matplotlib.
            xlog, ylog (bool): In log scale?
            au (bool): In au?
            Others: options for plot.

        Return:
            matplotlib ax.
        """

        # plot setting
        plt.rcParams['font.family']     = 'Arial'  # font (Times New Roman, Helvetica, Arial)
        plt.rcParams['xtick.direction'] = 'in'     # directions of x ticks ('in'), ('out') or ('inout')
        plt.rcParams['ytick.direction'] = 'in'     # directions of y ticks ('in'), ('out') or ('inout')
        plt.rcParams['font.size']       = fontsize # fontsize

        # output name
        if outname:
            pass
        else:
            outname = self.outname + '_pvfit_rvplane'

        # figure
        if ax:
            pass
        else:
            fig = plt.figure(figsize=(11.69, 8.27))
            ax  = fig.add_subplot(111)

        if colors:
            pass
        else:
            colors = {'ridge': {'red': 'red', 'blue':'blue'},
            'edge': {'red': 'pink', 'blue': 'skyblue'}}

        # plot fitting results
        if self.__sorted == False:
            self.sort_fitresults()

        # plot
        for i in self.results_sorted:
            for j in self.results_sorted[i]:
                #print (self.results_sorted[i][j])
                ax.errorbar(*self.results_sorted[i][j][0:2], xerr=self.results_sorted[i][j][2],
                    yerr=self.results_sorted[i][j][3], color=colors[i][j],
                    marker=marker, capsize=capsize, capthick=capthick, ls='')

        ax.tick_params(which='both', direction='in', bottom=True, top=True,
         left=True, right=True, pad=9)

        if xlog:
            ax.set_xscale('log')

        if ylog:
            ax.set_yscale('log')

        if len(xlim) == 2:
            ax.set_xlim(xlim[0], xlim[1])

        if len(ylim) == 2:
            ax.set_ylim(ylim[0], ylim[1])

        ax.set_xlabel(r'Radius (au)')
        ax.set_ylabel(r'Velocity (km s$^{-1}$)')

        plt.savefig(outname+'.'+outformat, transparent=True)
        plt.close()
        return ax



### functions
def kepler_mass(r, v, unit):
    return v**2 * np.abs(r) * unit

def kepler_mass_error(r, v, dr, dv, unit):
    return kepler_mass(r, v, unit) * np.sqrt((2*dv/v)**2 + (dr/r)**2)

def between(t, tlim):
    if not (len(tlim) == 2):
        return np.full(np.shape(t), True)
    else:
        return (tlim[0] < t) * (t < tlim[1])
