# -*- coding: utf-8 -*-

'''
Program to perform Gaussian fitting to a PV diagram.
Made and developed by J. Sai.

E-mail: jn.insa.sai@gmail.com

Latest update: 1/12/2020

Yusuke's note:
rng = range(1, naxis + 1) is introduced in read_pvfits.
res_ridge and res_edge are stored after being transposed.
Added multipeaks option in xcut as well just in case and for symmetry.
thrindx is not defined.
Variables not used are commented out.
The function "between" can receive tlim=[] now.
Please search with ##### for other minor comments.

'''


# modules
import copy
import numpy as np
import matplotlib.pyplot as plt
#import pandas as pd
#from astropy.io import fits
from astropy import constants, units

from .fitfuncs import edge, ridge_mean, gaussfit, gauss1d
from pvfit.pvfits import Impvfits
from pvfit.analysis_tools import doublepower_r, doublepower_v
from utils import emcee_corner


# Constants (in cgs)
Ggrav  = constants.G.cgs.value     # Gravitational constant
Msun   = constants.M_sun.cgs.value # Solar mass (g)
au     = units.au.to('cm')         # au (cm)
clight = constants.c.cgs.value     # light speed (cm s^-1)



class PVFit():
    """
    Perform fitting to a PV diagram.

    Parameters
    ----------
    infile(str): An input fits file.
    rms (float): rms noise level of the pv diagram.
    thr (int/float): threshold above which emission is used for fitting.
    """
    def __init__(self, infile, rms, vsys, dist, pa=None):
        """_summary_

        Args:
            infile (_type_): _description_
            rms (_type_): _description_
            vsys (_type_): _description_
            dist (_type_): _description_
            pa (_type_, optional): _description_. Defaults to None.
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
        pixrng_vcut=2, pixrng_xcut=None,
        Mlim=[0, 1e10], xlim=[-1e10, 0, 0, 1e10], vlim=[-1e10, 0, 0, 1e10],
        use_velocity=True, use_position=True,
        interp_ridge=False):
        """Get the edge/ridge at positions/velocities.

        Args:
            outname (_type_): _description_
            thr (_type_, optional): _description_. Defaults to 5..
            incl (_type_, optional): _description_. Defaults to 90..
            quadrant (_type_, optional): _description_. Defaults to None.
            mode (str, optional): _description_. Defaults to 'mean'.
            pixrng_vcut (int, optional): _description_. Defaults to 2.
            pixrng_xcut (_type_, optional): _description_. Defaults to None.
            Mlim (list, optional): _description_. Defaults to [].
            xlim (list, optional): _description_. Defaults to [].
            vlim (list, optional): _description_. Defaults to [].
            use_velocity (bool, optional): _description_. Defaults to True.
            use_position (bool, optional): _description_. Defaults to True.
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
            ##### np.squeeze()
            data = data[0,:,:] # Remove stokes I
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
        # Get rigde/edge
        if use_position:
            self.pvfit_xcut(outname, self.rms, vsys=self.vsys,
                            dist=self.dist, thr=thr, incl=incl,
                            xlim=xlim, vlim=vlim, Mlim=Mlim,
                            mode=mode, pixrng=pixrng_xcut)
        if use_velocity:
            self.pvfit_vcut(outname, self.rms, vsys=self.vsys,
                            dist=self.dist, thr=thr, incl=incl,
                            xlim=xlim, vlim=vlim, Mlim=Mlim,
                            mode=mode, pixrng=pixrng_vcut)
        # sort results
        self.sort_fitresults()
        # plot
        self.plotresults_pvdiagram()
        self.plotresults_rvplane()

    def sort_fitresults(self):
        '''
        Sort fitting results.
        '''
        # results are sorted?
        self.__sorted = True
        # to store
        self.results_sorted = {'ridge': {'red': None, 'blue': None},
                               'edge': {'red': None, 'blue': None}}
        self.results_filtered = {'ridge': None, 'edge': None}
        # sort results
        for i in ['ridge', 'edge']:
            # tentative space
            self.__store = {'xcut': {'red': None, 'blue': None},
                            'vcut': {'red': None, 'blue': None}}
            for j in ['xcut', 'vcut']:
                if self.results[i][j] is None:
                    continue
                # x, v, err_x, err_v
                results = copy.deepcopy(self.results[i][j])
                results[1] -= self.vsys
                # relative velocity to separate red/blue comp.
                vrel = results[1]
                xrel = results[0]
                # separate red/blue components
                ##### need confirmation. No abs now.
                rel = vrel if j == 'xcut' else xrel * self.xsign
                redsign = int(self.xsign) if j == 'vcut' else 1
                res_red  = [k[rel > 0][::redsign] for k in results]
                res_blue = [k[rel < 0][::-redsign] for k in results]
                ## nan after turn over
                ##### need confirmation
                jj = 0 if j == 'xcut' else 1
                for a in [res_red[jj], res_blue[jj]]:
                    if np.any(~np.isnan(a)):
                        a[:np.nanargmax(np.abs(a))] = np.nan 
                # remove points outside Mlim
                ##### need confirmation
                if len(self.Mlim) == 2:
                    for a in [res_red, res_blue]:
                        mass_est = kepler_mass(a[0], a[1], self.__unit)
                        a[jj][~between(mass_est, self.Mlim)] = np.nan
                self.__store[j]['red']  = res_red
                self.__store[j]['blue'] = res_blue
            # remove low-velocity positions and inner velocities when using both positions and velocities
            ##### need confirmation
            for j in ['red', 'blue']:
                x1, v0, _, _ = self.__store['xcut'][j]
                x0, v1, _, _ = self.__store['vcut'][j]
                xx, xv = np.meshgrid(x1, x0)
                vx, vv = np.meshgrid(v0, v1)
                xvd = np.hypot((xx - xv) / self.fitsdata.res_off,
                               (vx - vv) / self.fitsdata.delv)
                if np.any(~np.isnan(xvd)):
                    iv, ix = np.unravel_index(np.nanargmin(xvd), np.shape(xx))
                    x1[:ix] = np.nan
                    v1[:iv] = np.nan
                #xmax = np.nanmax(self.__store['xcut'][j][0])
                #vmax = np.nanmax(self.__store['vcut'][j][1])
                #self.__store['xcut'][j][0][self.__store['xcut'][j][1] < vmax] = np.nan
                #self.__store['vcut'][j][1][self.__store['vcut'][j][0] < xmax] = np.nan
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
                ##### need confirmation
                elif not ((self.results[i]['xcut'] is None)
                          and (self.results[i]['vcut'] is None)):
                    # only remove nan
                    if self.results[i]['xcut'] is not None:
                        cut, jj, res = 'xcut', 0, self.__store['xcut'][j]
                    else:
                        cut, jj, res = 'vcut', 1, self.__store['vcut'][j]
                    res_comb = [k[~np.isnan(res[jj])] for k in res]
                    res_f[cut][j] = [k[~np.isnan(res[jj])] for k in res]
                else:
                    print('ERROR\tsort_fitresults: '
                          + 'No fitting results are found.')
                    return

                # arcsec --> au
                ##### need confirmation. originally 0 and 3.
                res_comb[0] = res_comb[0]*self.dist
                res_comb[2] = res_comb[2]*self.dist
                for cut in ['xcut', 'vcut']:
                    res_f[cut][j][0] = res_f[cut][j][0]*self.dist
                    res_f[cut][j][2] = res_f[cut][j][2]*self.dist
                ## sort by x
                i_order  = np.argsort(np.abs(res_comb[0]))
                res_comb = np.array([np.abs(k[i_order]) for k in res_comb]).T
                # save
                self.results_sorted[i][j] = res_comb
            self.results_filtered[i] = res_f
        del self.__store

    # pvfit
    def pvfit_vcut(self, outname, rms, thr=5., vsys=0, dist=140.,
                   incl=90., xlim=[-1e10, 0, 0, 1e10],
                   vlim=[-1e10, 0, 0, 1e10], Mlim=[0, 1e10], mode='gauss',
                   pixrng=2, multipeaks=False, i_peak=0, prominence=1.5,
                   inverse=None, quadrant='13'):
        ##### self.dist and dist?
        '''
        Fitting along the velocity axis, i.e., fitting to the spectrum at each offset.

        Parameters
        ----------
         outname: output file name
         rms: rms noise level used to determine which data are used for the fitting.
         thr: Threshold for the fitting. Spectrum with the maximum intensity above thr x rms will be used for the fitting.
         xlim, vlim: x and v ranges for the fitting. Must be given as a list, [outlimit1, inlimit1, inlimit2, outlimit2].
         pixrng: Pixel range for the fitting around the maximum intensity. Only velocity channels +/- pixrng 
                  around the channel with the maximum intensity are used for the fitting.
         multipeaks: Find multiple peaks if True. Default False, which means only the maximum peak is considered.
         i_peak: Only used when multipeaks=True. Used to specify around which intensity peak you want to perform the fitting.
         prominence: Only used when multipeaks=True. Parameter to determine multiple peaks.
         inverse: Inverse axes. May be used to sort the sampling between fittings with different xlim or vlim.
        '''
        # modules
        import math
        #from scipy import optimize
        from scipy.signal import find_peaks
        from scipy.interpolate import interp1d
        from mpl_toolkits.axes_grid1 import ImageGrid

        print('pvfit along velocity axis.')
        # data
        data = self.fitsdata.data
        if not (self.fitsdata.naxis in [2, 3]):
            print('ERROR\tpvfit_vcut: n_axis must be 2 or 3.')
            return
        if self.fitsdata.naxis == 3:
            ##### np.squeeze()
            data = data[0,:,:] # Select stokes I
        # axes
        xaxis = self.fitsdata.xaxis
        vaxis = self.fitsdata.vaxis
        # resolution
        if self.fitsdata.res_off:
            res_off = self.fitsdata.res_off
        else:
            bmaj, bmin, bpa = self.fitsdata.beam
            ##### why bmin?
            res_off = bmin # in arcsec
        # harf of beamsize [pix]
        hob = int(np.round((res_off*0.5/self.fitsdata.delx)))

        # relative velocity or LSRK
        #xlabel = 'Offset (arcsec)'
        #vlabel = r'LSR velocity ($\mathrm{km~s^{-1}}$)'

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
            data_fit  = np.array([d[b] for d in data.T]).T
            vmin, vmax = np.min(vaxis_fit), np.max(vaxis_fit)
            print(f'v range: {vmin:.2f} -- {vmax:.2f} km/s')
        # to achieve the same sampling on two sides
        ##### need to confirm the meaning of inverse.
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
        axes_pad=0,share_all=True, aspect=False, label_mode='L') #,cbar_mode=cbar_mode)
        #gridi = 0
        dlim  = [np.nanmin(data_fit), np.nanmax(data_fit)]
        # x & y label
        grid[(nrow*ncol - ncol)].set_xlabel(r'Velocity (km s$^{-1}$)')
        grid[(nrow*ncol - ncol)].set_ylabel('Intensity')
        # list to save final results
        ##### need confirmation
        res_ridge = np.empty((nloop, 4))
        res_edge  = np.empty((nloop, 4))
        #res_ridge = np.zeros((nloop, 4)) # x, v, err_x, err_v
        #res_edge  = np.zeros((nloop, 4))
        # loop for x
        for i in range(nloop):
            # ith data
            x_i = xaxis_fit[i]
            d_i = data_fit[:, i]
            if np.all(np.isnan(d_i)) or (xlim[1] < x_i < xlim[2]):
                res_ridge[i, :] = [x_i, np.nan, 0, np.nan]
                res_edge[i, :] = [x_i, np.nan, 0, np.nan]
                continue
            # plot results
            ##### need confirmation
            ax = grid[i]
            #print('x_now: %.2f arcsec'%x_i)
            snr = np.nanmax(d_i) / rms
            posacc = res_off / snr
            # get ridge value
            if mode == 'mean':
                c = (d_i >= thr * rms)
                v_i, d_i = vaxis_fit[c], d_i[c]
                mv, mv_err = ridge_mean(v_i, d_i, rms)
                # plot
                if ~np.isnan(mv):
                    ax.vlines(mv, 0., dlim[-1], lw=1.5,
                              color='r', ls='--', alpha=0.8)
            elif mode == 'gauss':
                # get peak indices
                peaks, _ = find_peaks(d_i, height=thr * rms,
                                      prominence=prominence * rms)
                #print(peaks)
                #if len(peaks) > 0:
                #    snr = np.nanmax(d_i) / rms
                #    posacc = res_off / snr
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
                        # peak position is too edge
                        mv, mv_err = np.nan, np.nan
                    else:
                        # use pixels only around intensity peak
                        v0, v1 = pidx - pixrng, pidx + pixrng + 1
                        d_i, v_i  = d_i[v0:v1], vaxis_fit[v0:v1]
                        #nd_i = len(d_i)
                else:
                    v_i = vaxis_fit.copy()
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
            # output ridge results
            res_ridge[i, :] = [x_i, mv, posacc, mv_err]

            # get edge values
            # fitting axis
            v_i = vaxis_fit.copy()
            d_i = data_fit[:,i]
            # interpolation
            vi_interp  = np.linspace(v_i[0], v_i[-1],
                                     (len(v_i)-1)*10 + 1) # resample with a 10 times finer sampling rate
            di_interp  = interp1d(v_i, d_i, kind='cubic')(vi_interp)           # interpolation
            # flag by mass
            mass_est = kepler_mass(x_i, vi_interp-vsys, self.__unit)
            goodflag = between(mass_est, Mlim) if len(Mlim) == 2 else None
            edgesign = v_i[np.nanargmax(d_i)] - vsys
            mv, mv_err = edge(vi_interp, di_interp, rms, rms*thr,
                              goodflag=goodflag, edgesign=edgesign)
            if not (vlim[0] < mv < vlim[1] or vlim[2] < mv < vlim[3]):
                mv, mv_err = np.nan, np.nan
            res_edge[i, :] = [x_i, mv, posacc, mv_err]
            # plot
            if ~np.isnan(mv):
                ax.vlines(mv, 0., dlim[-1], lw=1.5, color='b',
                          ls='--', alpha=0.8)
            # observed data
            ax.step(vaxis_fit, data_fit[:,i], linewidth=1.,
                    color='k', where='mid')
            # offset label
            ax.text(0.9, 0.9, f'{x_i:03.2f}', horizontalalignment='right',
                verticalalignment='top',transform=ax.transAxes)
            ax.tick_params(which='both', direction='in',bottom=True,
                           top=True, left=True, right=True, pad=9)
        # Store the result array in the shape of (4, len(x)).
        ##### need confirmation
        self.results['ridge']['vcut'] = res_ridge.T
        self.results['edge']['vcut']  = res_edge.T
        #plt.show()
        fig.savefig(outname + "_pvfit_vcut.pdf", transparent=True)
        plt.close()
        ### output results as .txt file
        #res_hd  = 'offset[arcsec]\tvelocity[km/s]\toff_err[arcesc]\tvel_err[km/s]'
        #res_all = np.c_[res_x, res_v, err_x, err_v]
        #np.savetxt(outname+'_pv_vcut.txt', res_all,fmt = '%4.4f', delimiter ='\t', header = res_hd)


    def pvfit_xcut(self, outname, rms, thr=5., vsys=0, dist=140.,
                   incl=90., xlim=[-1e10, 0, 0, 1e10],
                   vlim=[-1e10, 0, 0, 1e10], Mlim=[0, 1e10], mode='mean',
                   pixrng=None, multipeaks=False, i_peak=0,
                   prominence=1.5):
        '''
        Fitting along x-axis, i.e., fitting to the intensity distribution at each velocity.

        Parameters
        ----------
         outname: Outputname
         rms: rms noise level used to determine which data are used for the fitting.
         thr: Threshold for the fitting. Spectrum with the maximum intensity above thr x rms will be used for the fitting.
         xlim, vlim: x and v ranges for the fitting. Must be given as a list, [minimum, maximum].
         pixrng: Pixel range for the fitting around the maximum intensity. Only velocity channels +/- pixrng 
                  around the channel with the maximum intensity are used for the fitting.
        '''
        # modules
        import math
        from scipy import optimize
        from scipy.signal import find_peaks
        from scipy.interpolate import interp1d
        from mpl_toolkits.axes_grid1 import ImageGrid

        print('pvfit along position axis.')

        # data
        data = self.fitsdata.data
        if not (self.fitsdata.naxis in [2, 3]):
            print('Error\tpvfit_vcut: n_axis must be 2 or 3.')
            return
        elif self.fitsdata.naxis == 3:
            data = data[0,:,:] # Remove stokes I
        # axes
        xaxis = self.fitsdata.xaxis
        vaxis = self.fitsdata.vaxis
        # resolution
        if self.fitsdata.res_off:
            res_off = self.fitsdata.res_off
        else:
            bmaj, bmin, bpa = self.fitsdata.beam
            res_off = bmin  # in arcsec
        # harf of beamsize [pix]
        hob = int(np.round((res_off*0.5/self.fitsdata.delx)))
        delv = self.fitsdata.delv
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
            data_fit  = np.array([d[b] for d in data.T]).T
            vmin, vmax = np.min(vaxis_fit), np.max(vaxis_fit)
            print(f'v range: {vmin:.2f} -- {vmax:.2f} km/s')
        # Nyquist sampling
        ##### need for xcut fit, too?
        xaxis_fit = xaxis[::hob]
        vaxis_fit = vaxis
        data_fit  = data[:,::hob]
        nloop = len(vaxis_fit)
        ncol = int(math.ceil(np.sqrt(nloop)))
        nrow = int(math.ceil(nloop / ncol))
        # figure for check result
        fig  = plt.figure(figsize=(11.69, 8.27))
        grid = ImageGrid(fig, rect=111, nrows_ncols=(nrow,ncol),
        axes_pad=0,share_all=True, aspect=False, label_mode='L') #,cbar_mode=cbar_mode)
        #gridi = 0
        # x & y label
        grid[(nrow*ncol-ncol)].set_xlabel('Offset (arcsec)')
        grid[(nrow*ncol-ncol)].set_ylabel('Intensity')
        dlim = [np.nanmin(data_fit), np.nanmax(data_fit)]
        # list to save final results
        ##### need confirmation
        res_ridge = np.empty((nloop, 4))
        res_edge  = np.empty((nloop, 4))
        #res_ridge = np.zeros((nloop, 4)) # x, v, err_x, err_v
        #res_edge  = np.zeros((nloop, 4))
        # loop for v
        for i in range(nloop):
            # ith data
            v_i = vaxis_fit[i]
            d_i  = data_fit[i, :]
            if np.all(np.isnan(d_i)) or (vlim[1] < v_i < vlim[2]):
                res_ridge[i, :] = [np.nan, v_i, np.nan, 0]
                res_edge[i, :] = [np.nan, v_i, np.nan, 0]
                continue
            # plot results
            ##### need confirmation
            ax = grid[i]
            #print('Channel #%i velocity %.2f km/s'%(i+1, v_i))
            snr = np.nanmax(d_i) / rms
            velacc = delv / snr
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
                #print(peaks)
                #if len(peaks) > 0:
                #    snr = np.nanmax(d_i)/rms
                #    velacc = delv / snr
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
                print('ERROR\tpvfit_vcut: mode must be mean or gauss.')
                return
            # output ridge results
            if not (xlim[0] < mx < xlim[1] or xlim[2] < mx < xlim[3]):
                mx, mx_err = np.nan, np.nan
            ##### 1/sqrt(12) comes from sqrt(int^(1/2)_(-1/2)x**2dx)?
            #res_ridge[i, :] = [mx, v_i, mx_err, delv / (2. * np.sqrt(3))]
            ##### need confirmation
            res_ridge[i, :] = [mx, v_i, mx_err, velacc]

            # get edge values
            # fitting axes
            x_i = xaxis_fit.copy()
            d_i = data_fit[i, :]
            # interpolation
            xi_interp  = np.linspace(x_i[0], x_i[-1],
                                     (len(x_i)-1)*10 + 1) # resample with a 10 times finer sampling rate
            di_interp  = interp1d(x_i, d_i, kind='cubic')(xi_interp)           # interpolation
            # flag by mass
            mass_est = kepler_mass(xi_interp, v_i - vsys, self.__unit)
            goodflag = between(mass_est, Mlim) if len(Mlim) == 2 else None
            edgesign = x_i[np.nanargmax(d_i)]
            mx, mx_err = edge(xi_interp, di_interp, rms, rms*thr,
                              goodflag=goodflag, edgesign=edgesign)
            if not (xlim[0] < mx < xlim[1] or xlim[2] < mx < xlim[3]):
                mx, mx_err = np.nan, np.nan
            #res_edge[i, :] = [mx, v_i, mx_err, delv / (2. * np.sqrt(3))]
            ##### for confirmation
            res_edge[i, :] = [mx, v_i, mx_err, velacc]
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
                      show_corner: bool = False,
                      minrelerr: float = 0.01,
                      minabserr: float = 0.1):
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
        def clipped_error(err, val, mode):
            res = self.fitsdata.beam[0] if mode == 'x' else self.fitsdata.delv
            minabs = [minabserr * res] * len(err)
            return np.max([err, minrelerr * np.abs(val), minabs], axis=0)
    
        res_org = self.results_filtered
        Ds = [None, None]
        for i, er in enumerate(['edge', 'ridge']):
            xrb = [res_org[er]['xcut'][rb] for rb in ['red', 'blue']]
            xcut = np.concatenate(xrb, axis=1)
            v0, x1, dx1 = xcut[1], self.xsign * xcut[0], xcut[2]
            dx1 = clipped_error(dx1, x1, mode='x')
            vrb = [res_org[er]['vcut'][rb] for rb in ['red', 'blue']]
            vcut = np.concatenate(vrb, axis=1)
            x0, v1, dv1 = self.xsign * vcut[0], vcut[1], vcut[3]
            dv1 = clipped_error(dv1, v1, mode='v')
            Ds[i] = [v0, x1, dx1, x0, v1, dv1]
        Es, Rs = Ds
        
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
            result = self.results_filtered[re]
            xmax = np.max([[np.max(np.abs(result[xv][rb][0]))
                                for xv in ['xcut', 'vcut']]
                           for rb in ['red', 'blue']])
            xmin = np.min([[np.min(np.abs(result[xv][rb][0]))
                                for xv in ['xcut', 'vcut']]
                           for rb in ['red', 'blue']])
            if loglog:
                s, x = 1, np.geomspace(xmin, xmax, 100)
            else:
                s, x = self.xsign, np.linspace(-xmax, xmax, 100)
                x[(-xmin < x) * (x < xmin)] = None
            x, y = x, s * model[re](x)
            if flipaxis: x, y = y, x
            ax.plot(x, y, ls=ls[re], lw=2, color='gray', zorder=3)
            
        
    def plawfit(self, outname, params_inp, mode='sp',
     inner_threshold=None, outer_threshold=None, dist=140., vsys=0.):
        '''
        Fit a power-law function.

        '''

        # modules
        import os
        from scipy import optimize
        from scipy.stats import norm, uniform
        from matplotlib.gridspec import GridSpec

        # functions
        def splaw(r, params, r0=100.):
            '''
            Single power-law function.

            Args:
             - r: radius (au or any)
             - params: [vsys, v0, p]
             - r0: 100 (au)
            '''
            vsys, v0, p = params
            vout        = v0*(r/r0)**(-p)
            dydx        = (v0/r0)*(-p)*(r/r0)**(-p-1)
            return vout, dydx

        def chi_splaw(params, xdata, ydata, xsig, ysig):
            '''
            Calculate chi for chi-square for the single power-law function.
            '''
            vsys       = params[0]
            vout, dydx = splaw(xdata, params)
            #sig       = np.sqrt((xsig*dydx)*(xsig*dydx) + ysig*ysig)
            sig        = ysig
            chi_out    = (np.abs(ydata - vsys) - vout)/sig
            return chi_out


        def dplaw(radii, params):
            '''
            Double power-law function.

            Args:
             - r: radius (au or any)
             - params: [vb, rb, pin, pout]
              - vb: rotational velocity at rb [km/s]
              - rb: break point raidus at which powers change [au]
              - pin: power at r < rb
              - pout: power at r >= rb
            '''
            vb, rb, pin, pout = params

            vout = np.array([
                vb*(r/rb)**(-pin) if r < rb else vb*(r/rb)**(-pout)
                for r in radii
                ])
            dydx = np.array([
                (vb/rb)*(-pin)*(r/rb)**(-pin-1) if r < rb else (vb/rb)*(-pout)*(r/rb)**(-pout-1)
                for r in radii
                ])

            return vout, dydx


        def chi_dplaw(params, xdata, ydata, xsig, ysig):
            '''
            Chi for chi square for the double power-law function.

            Args:
             - func: a function.
             - params: input parameters for the function.
             - xdata: x of data
             - ydata: y of data
             - xsig: sigma for x
             - ysig: sigma for y
            '''
            vout, dydx = dplaw(xdata, params)
            #sig       = np.sqrt((xsig*dydx)*(xsig*dydx) + ysig*ysig)
            sig        = ysig
            chi_out    = (np.abs(ydata - vout))/sig
            return chi_out


        def estimate_perror(params, func, x, y, xerr, yerr, niter=3000):
            '''
            Estimate fitting-parameter errors by Monte-Carlo method.

            '''
            nparams = len(params)
            perrors = np.zeros((0,nparams), float)

            for i in range(niter):
                offest = norm.rvs(size = len(x), loc = x, scale = xerr)
                velest = norm.rvs(size = len(y), loc = y, scale = yerr)
                result = optimize.leastsq(func, params, args=(offest, velest, xerr, yerr), full_output = True)
                perrors = np.vstack((perrors, result[0]))
                #print param_esterr[:,0]

            sigmas = np.array([
                np.std(perrors[:,i]) for i in range(nparams)
                ])
            medians = np.array([
                np.median(perrors[:,i]) for i in range(nparams)
                ])


            with np.printoptions(precision=4, suppress=True):
                print ('Estimated errors (standard deviation):')
                print (sigmas)
                print ('Medians:')
                print (medians)


            # plot the Monte-Carlo results
            fig_errest = plt.figure(figsize=(11.69,8.27), frameon = False)
            gs         = GridSpec(nparams, 2)
            xlabels    = [r'$V_\mathrm{sys}$', r'$V_\mathrm{100}$', r'$p$' ]
            for i in range(nparams):
                # histogram
                ax1 = fig_errest.add_subplot(gs[i,0])
                ax1.set_xlabel(xlabels[i])
                if i == (nparams - 1):
                    ax1.set_ylabel('Frequency')
                ax1.hist(perrors[:,i], bins = 50, cumulative = False) # density = True

                # cumulative histogram
                ax2 = fig_errest.add_subplot(gs[i,1])
                ax2.set_xlabel(xlabels[i])
                if i == (nparams - 1):
                    ax2.set_ylabel('Cumulative\n frequency')
                ax2.hist(perrors[:,i], bins = 50, density=True, cumulative = True)

            plt.subplots_adjust(wspace=0.4, hspace=0.4)
            #plt.show()
            fig_errest.savefig(outname + '_errest.pdf', transparent=True)
            fig_errest.clf()

            return sigmas



        # ----------- !!start!! ---------
        if len(self.results) == 0:
            print ('No fitting result is found. Run pvfit or read_results before.')
            return
        else:
            results = self.results['0']
            for i in range(1, self.dicindx):
                results = np.hstack([results, self.results['%i'%i]])

        offset, velocity, off_err, vel_err, chi2_pvfit = results

        # Prevent dtype-error
        offset     = offset.astype('float64')
        velocity   = velocity.astype('float64')
        off_err    = off_err.astype('float64')
        vel_err    = vel_err.astype('float64')
        chi2_pvfit = chi2_pvfit.astype('float64')

        # -offset --> offset
        offset  = np.abs(offset)*dist
        off_err = off_err*dist


        #if inner_threshold:
        #    indx     = np.where(np.abs(offset) >= inner_threshold)
        #    offset   = offset[indx]
        #    velocity = velocity[indx]
        #    off_err  = off_err[thrindx]
        #    vel_err  = vel_err[thrindx]

        #if outer_threshold:
        #    indx     = np.where(np.abs(offset) >= outer_threshold)
        #    offset   = offset[indx]
        #    velocity = velocity[indx]
        #    off_err  = off_err[thrindx]
        #    vel_err  = vel_err[thrindx]

        if mode == 'sp':
            func     = splaw
            func_chi = chi_splaw
        elif mode == 'dp':
            func     = dplaw
            func_chi = chi_dplaw
            velocity = np.abs(velocity - vsys)
        else:
            print ('Error: mode must be sp or dp.')
            return

        # fitting
        ndata        = len(offset)
        nparams      = len(params_inp)
        result       = optimize.leastsq(func_chi, params_inp, args=(offset, velocity, off_err, vel_err), full_output = True)
        params_out   = result[0]
        chi          = func_chi(params_out, offset, velocity, off_err, vel_err)
        chi2         = np.sum(chi*chi)
        dof          = ndata - nparams # -1
        reduced_chi2 = chi2/dof

        if mode == 'sp':
            vsys = params_out[0]

        # output
        print ('Results:')
        #print 'chi2', chi2
        print ('')
        with np.printoptions(precision=4, suppress=True):
            print (params_out)
        print ('reduced chi^2: %.2f'%reduced_chi2)
        #print ('degree of freedum: %.f'%DOF)

        # error
        perrors = estimate_perror(params_out, func_chi, offset, velocity, off_err, vel_err, niter=3000)

        self.params_out = params_out
        self.perrors    = perrors


        # plot results
        ax = self.plotresults_onrvplane(vsys=vsys, au=True)

        rmodel = np.logspace(0,4,128)
        vmodel = func(rmodel, params_out)[0]
        ax.plot(rmodel, vmodel, color='k', lw=2., alpha=0.8, ls='-')

        ax.set_xlim(np.nanmin(offset)*0.9, np.nanmax(offset)*1.1)
        ax.set_ylim(np.nanmin(np.abs(velocity-vsys))*0.9, np.nanmax(np.abs(velocity-vsys))*1.1)

        ax.set_xlabel(r'Radius (au)')
        ax.set_ylabel(r'Velocity ($\mathrm{km~s^{-1}}$)')

        plt.savefig(outname + '_plawfitres.pdf', transparent=True)
        return ax



    # Plot results
    def plotresults_pvdiagram(self, outname=None, marker='o', colors=['r'], alpha=1.,
        data=None,header=None,ax=None,outformat='pdf',pvcolor=True,cmap='Greys',
        vmin=None,vmax=None, contour=True,clevels=None,pvccolor='k',
        vrel=False,logscale=False,x_offset=False,ratio=1.2, prop_vkep=None,fontsize=14,
        lw=1,clip=None,plot_res=True,inmode='fits',xranges=[], yranges=[],
        ln_hor=True, ln_var=True, pvalpha=None):
        '''
        Plot fitting results on a PV diagram.


        Parameters
        ----------
         outname: Output file name.
         outformat: Format of the output file. Deafult pdf.
         marker: Marker for the data points.
         colors: Colors for the data points. Given as a list. If PVFit object has multiple results, more than one colors can be given.
         alpha: Transparency alpha for the plot.
         ax: Axis of python matplotlib.
         pvcolor: Plot PV diagram in color scale? Default True.
         cmap: Color map.
         vmin, vmax: Minimum and maximum values for the color scale.
         vsys: Plot the systemic velocity with a dashed line if given.
         contour: Plot a PV diagram with contour? Default True.
         clevels: Contour levels. Given in absolute intensity.
         pvccolor: Contour color.
         pa: Position angle of the PV diagram. Used to calculate angular resolution along the PV slice if given.
         vrel: Vaxis will be in the relative velocity with respect to vsys if True.
         x_offset: xaxis will be offset axis if True. Default False, which means xaxis is velocity axis.
         ratio: Aspect ration of the plot.
         lw: Line width of the contour.
         inmode: Data and header information can be given with parameters of data and header if inmode=data. Default fits.
                  This option is used when you want to modify data before plot (e.g., multipy a factor to data).
         data: Data of PV diagram. Must be given as an array.
         header: Header information. Only used when inmode=data.
         xranges, yranges: x and y ranges for plot. Given as a list, [minimum, maximum].
        '''
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
        ax = self.fitsdata.draw_pvdiagram(outname, data=data, header=header, ax=ax,
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
                results[i]['red'].T[0]  *= self.xsign/self.dist
                results[i]['blue'].T[0] *= -self.xsign/self.dist
                results[i]['red'].T[3]  *= self.xsign/self.dist
                results[i]['blue'].T[3] *= -self.xsign/self.dist

                results[i]['red'].T[1]  = self.vsys + results[i]['red'].T[1]
                results[i]['blue'].T[1] = self.vsys - results[i]['blue'].T[1]

                colors = {'ridge': {'red': 'red', 'blue':'blue'},
                'edge': {'red': 'pink', 'blue': 'skyblue'}}

                for j in results[i]:
                    ax = plotpoints(ax, *results[i][j].T, colors[i][j], x_offset)
        else:
            results = self.results

            colors = {'ridge':'r', 'edge':'b'}
            for i in results:
                #color = colors[i]
                if self.results[i]['vcut'] is not None:
                    ax = plotpoints(ax, *self.results[i]['vcut'].T, colors[i], x_offset)

                if self.results[i]['xcut'] is not None:
                    ax = plotpoints(ax, *self.results[i]['vcut'].T, colors[i], x_offset)

        plt.savefig(outname+'.'+outformat, transparent=True)
        plt.close()
        return ax


    def plotresults_rvplane(self, outname=None, outformat='pdf', ax=None,
        xlog=True, ylog=True, au=False, marker='o',
        capsize=2, capthick=2., colors=None, xlim=[], ylim=[], fontsize=14):
        '''
        '''

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
            fig = plt.figure(figsize=(8.27, 8.27))
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
                ax.errorbar(*self.results_sorted[i][j].T, color=colors[i][j],
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
