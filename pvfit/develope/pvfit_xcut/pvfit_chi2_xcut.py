
# coding: utf-8
### Making r vs v diagram

### import modules
import os
import sys
import math
import subprocess
import numpy as np
from astropy.io import fits
import matplotlib.pyplot as plt
from astropy.wcs import WCS
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
from scipy import optimize
from scipy import stats
from matplotlib.offsetbox import AnchoredText
from mpl_toolkits.axes_grid1 import ImageGrid
#MCMC
import scipy.optimize as op
import emcee
import corner



### setting for figures
#mpl.use('Agg')
#mpl.rcParams['agg.path.chunksize'] = 100000
plt.rcParams['font.family'] = 'Arial'    # font (Times New Roman, Helvetica, Arial)
plt.rcParams['xtick.direction'] = 'in'  # directions of x ticks ('in'), ('out') or ('inout')
plt.rcParams['ytick.direction'] = 'in'  # directions of y ticks ('in'), ('out') or ('inout')
#plt.rcParams['xtick.major.width'] = 1.0 # x ticks width
#plt.rcParams['ytick.major.width'] = 1.0 # y ticks width
plt.rcParams['font.size'] = 11           # fontsize
#plt.rcParams['axes.linewidth'] = 1.0    # edge linewidth




### setting constants & parameters
clight = 2.99792458e10 # cm/s




### defining functions
def change_aspect_ratio(ax, ratio):
    '''
    This function change aspect ratio of figure.
    Parameters:
        ax: ax (matplotlit.pyplot.subplots())
            Axes object
        ratio: float or int
            relative x axis width compared to y axis width.
    '''
    aspect = (1/ratio) *(ax.get_xlim()[1] - ax.get_xlim()[0]) / (ax.get_ylim()[1] - ax.get_ylim()[0])
    aspect = np.abs(aspect)
    aspect = float(aspect)
    ax.set_aspect(aspect)


# calculate error propagation of dividing
def err_frac(denm,numr,sigd,sign):
    # denm: denominator (bunbo)
    # numr: numerator (bunshi)
    # sigd: error of denm
    # sign: error of numr
    err = np.sqrt((sign/denm)*(sign/denm) + (numr*sigd/(denm*denm))*(numr*sigd/(denm*denm)))
    return err


# weighted center
def weighted_center(x,Ix):
    '''
    Derive weighted center.

    x: position
    Ix: Intensity at x
    '''
    xIx = x*Ix
    xmean = np.sum(xIx)/np.sum(Ix)

    return xmean


# gaussian
def gaussian(x,A,mean,sig):
    coeff=A/(np.sqrt(2.0*np.pi)*sig)
    exp=np.exp(-(x-mean)*(x-mean)/(2.0*sig*sig))
    gauss=coeff*exp
    return(gauss)


#def fit_error()
# y-f(x)/yerr, contents of  xi square
def chi(param, xdata, ydata, ysig):
    A, mean, sigma = param
    chi = (ydata - (gaussian(xdata, A, mean, sigma))) / ysig
    return(chi)



# --------------- main: PV fitting --------------------
def pvfit(self, outname, rms, thr=None, offbgn=None, offend=None,
    velbgn=None, velend=None, figsave=False, vsys=None, pixrng = 2):

    ###  reading fits files
    data, header=fits.getdata(self,header=True)
    print 'fits data shape: ',data.shape



    ## setting
    # axes
    # header info.
    naxis      = int(header['NAXIS'])
    noff       = int(header['NAXIS1'])
    nvel       = int(header['NAXIS2'])
    bmaj       = header['BMAJ']*60.*60. # in arcsec
    bmin       = header['BMIN']*60.*60. # in arcsec
    bpa        = header['BPA']          # in degree
    bsize      = (bmaj + bmin)*0.5      # in arcsec

    offlabel   = header['CTYPE1']
    vellabel   = header['CTYPE2']
    thirdlabel = header['BUNIT']
    offunit    = header['CUNIT1']
    restfreq   = header['RESTFRQ'] # Hz
    refval_off = header['CRVAL1']  # in arcsec
    refval_vel = header['CRVAL2']
    refval_vel = clight*(restfreq - refval_vel)/restfreq # Hz --> radio velocity [cm/s]
    refval_vel = refval_vel*1.e-5         # cm/s --> km/s
    refpix_off = int(header['CRPIX1'])
    refpix_vel = int(header['CRPIX2'])
    del_off    = header['CDELT1']  # in arcsec
    del_vel    = header['CDELT2']
    del_vel    = - clight*del_vel/restfreq # Hz --> cm/s
    del_vel    = del_vel*1.e-5             # cm/s --> km/s
    #print refval_vel,del_off
    #print round(crvl_vel, 4)
    #print round(refval_vel - del_vel*34.,4)


    # sampling step; half of beam
    hob = np.round((bsize*0.5/del_off), 0) # harf of beamsize [pix]

    # check unit
    if offunit == 'degree' or offunit == 'deg':
        refval_off = refval_off*60.*60.
        del_off    = del_off*60.*60.


    # relative velocity or LSRK
    offlabel = 'offset (arcsec)'
    vellabel = 'LSRK velocity (km/s)'


    # set extent of an image
    offmin = refval_off + (1 - refpix_off)*del_off     # refpix is not in 0 start
    offmax = refval_off + (noff - refpix_off)*del_off  # delpix (n - 1)
    velmin = refval_vel + (1 - refpix_vel)*del_vel
    velmax = refval_vel + (nvel - refpix_vel)*del_vel


    # set axes
    offaxis = np.arange(offmin, offmax+del_off, del_off)
    velaxis = np.arange(velmin, velmax+del_vel, del_vel)
    #print offaxis
    #print len(velaxis), nvel


    # list to save final results
    res_vel = np.array([])
    res_off = np.array([])
    err_off = np.array([])
    err_vel = np.array([])



    # cutting at an velocity and determining the position

    # handle ranges used for calculation
    if offbgn:
        pass
    else:
        offbgn = 0

    if offend:
        pass
    else:
        offend = noff-1

    if velbgn:
        pass
    else:
        velbgn = 0

    if velend:
        pass
    else:
        velend = nvel-1

    npanel = velend - velbgn
    nrow = math.ceil(np.sqrt(npanel))
    nrow = int(nrow)
    ncol = nrow


    # figure for check result
    fig  = plt.figure(figsize=(11.69, 8.27))
    grid = ImageGrid(fig, rect=111, nrows_ncols=(nrow,ncol),
        axes_pad=0,share_all=True, aspect=False, label_mode='L') #,cbar_mode=cbar_mode)
    gridi = 0

    # x & y label
    grid[(nrow*ncol-ncol)].set_xlabel('offset (arcsec)')
    grid[(nrow*ncol-ncol)].set_ylabel(r'intensity ($\mathrm{Jy\ beam^{-1}}$)')


    # calculation
    for n_row in xrange(velbgn,velend):
        # where it is now!!!
        velnow = velaxis[n_row]
        print 'channel: ', n_row, ', velocity: ', velnow, ' km/s'

        # set offset range used
        Icalc = data[0, n_row, offbgn:offend]
        offcalc   = offaxis[offbgn:offend]
        ncalc     = len(Icalc)

        # using data points >= thr * sigma
        if thr:
            if np.nanmax(Icalc) < thr*rms:
                print 'SNR < ', thr
                print 'skip this point'
                continue

        # Nyquist sampling (with a step of half of beamsize)
        offcalc = offcalc[[int(i+hob*0.5) for i in xrange(int(ncalc-hob*0.5)) if i%hob==0]]
        Icalc   = Icalc[[int(i+hob*0.5) for i in xrange(int(ncalc-hob*0.5)) if i%hob==0]]

        # skip if pixel number for calculation is below 3
        if Icalc.size <= 3 or offcalc.size <= 3:
            gridi = gridi + 1
            continue
        #print hob*del_off, offcalc[1]-offcalc[0]


        # set the initial parameter
        sigx = 0.25
        mx   = offcalc[np.argmax(Icalc)]
        amp  = np.nanmax(Icalc)*np.sqrt(2.*np.pi)*sigx
        pinp = [amp, mx, sigx]


        # fitting
        results = optimize.leastsq(chi, pinp, args=(offcalc, Icalc, rms), full_output=True)

        # recording output results
        # param_out: fitted parameters
        # err: error of fitted parameters
        # chi2: chi-square
        # DOF: degree of freedum
        # reduced_chi2: reduced chi-square
        param_out    = results[0]
        param_cov    = results[1]
        chi2         = np.sum(chi(param_out, offcalc, Icalc, rms)**2.)
        ndata        = len(offcalc)
        nparam       = len(param_out)
        DOF          = ndata - nparam
        reduced_chi2 = chi2/DOF

        if (DOF > 0) and (param_cov is not None):
            param_cov = param_cov*reduced_chi2
        else:
            param_cov = np.inf

        # best fit value
        amp_fit, mx_fit, sigx_fit = param_out

        # fitting error
        param_err = []
        for j in xrange(nparam):
            try:
                param_err.append(np.abs(param_cov[j][j])**0.5)
            except:
                print 'Cannot calculate covariance. Parameter error will be set as 0.'
                error.append(0.00)
        amp_fit_err, mx_fit_err, sigx_fit_err = param_err

        # output results
        res_off = np.append(res_off, mx_fit)
        res_vel = np.append(res_vel, velnow)
        err_off = np.append(err_off, mx_fit_err)
        err_vel = np.append(err_vel, del_vel*0.5)
        #print 'offmean: ', offmean, ' arcsec'
        #print 'mean error: ', offm_err, ' arcsec'


        # print results
        print 'Chi2: ', reduced_chi2
        print 'Fitting results: amp   mean   sigma'
        print amp_fit, mx_fit, sigx_fit
        print 'Errors'
        print amp_fit_err, mx_fit_err, sigx_fit_err



        ### plot results
        ax = grid[gridi]

        # observed data
        #ax.step(offcalc, Icalc,'k-', where='mid',lw=1)
        ax.errorbar(offcalc, Icalc, yerr=rms, fmt= 'k.', color = 'k', capsize = 2, capthick = 1)

        # fitted Gaussian
        off_forgauss = np.linspace(np.nanmin(offcalc), np.nanmax(offcalc), 256) # offset axis for plot
        gauss = gaussian(off_forgauss, amp_fit, mx_fit, sigx_fit)
        ax.plot(off_forgauss, gauss, lw=1, color='r', ls='-')

        # velcoity label
        ax.text(0.9, 0.9, '%03.2f'%velnow, horizontalalignment='right',
            verticalalignment='top',transform=ax.transAxes)



        # step
        gridi = gridi + 1


    #plt.show()
    fig.savefig(outname+"_pv_xfit.pdf", transparent=True)
    plt.clf()


    ### output results as .txt file
    res_hd  = '\t offset[arcsec]\t velocity[km/s]\t off_err[arcesc]\t vel_err[km/s]'
    res_all = np.c_[res_off, res_vel, err_off, err_vel]
    np.savetxt(outname+'_pv_xfit.txt', res_all,fmt = '%04.4f', delimiter ='\t', header = res_hd)



    if figsave:
        # set figure
        fig  = plt.figure(figsize=(11.69, 8.27))
        ax   = fig.add_subplot(111)

        # set axes
        redata  = np.rot90(data[0,:,:])
        res_off = -res_off
        extent  = (velmin,velmax,offmin,offmax)
        xlabel  = vellabel
        ylabel  = offlabel


        # plot images
        clevels = np.array([-3., 3., 5., 7., 9., 11., 13., 15., 20., 25., 30., 40., 50., 60., 70.])*rms
        imcont  = ax.contour(redata, colors='k', origin='lower',extent=extent, levels=clevels, linewidths=1)
        ax.errorbar(res_vel, res_off, xerr=err_vel, yerr=err_off, marker='o', color='royalblue', capsize=1)

        # axes
        ax.set_xlim(velmin,velmax)
        ax.set_ylim(offmin,offmax)


        # axis labels
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)


        # lines showing offset 0 and relative velocity 0
        xline = plt.hlines(0, velmin, velmax, "black", linestyles='dashed', linewidths = 0.5)
        if vsys:
            yline = plt.vlines(vsys, offmin, offmax, "black", linestyles='dashed', linewidths = 0.5)
        ax.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True)


        # aspect ratio
        change_aspect_ratio(ax, 1.2)

        # save figure
        fig.savefig(outname+'_pv.pdf', transparent=True)

        return ax

    return
# -----------------------------------------------------------
# -----------------------------------------------------------


# input
if __name__ == '__main__':
    ### input parameters
    imagename = '../../../l1489.c18o.contsub.gain01.rbp05.mlt100.cf15.pbcor.pv69d.fits'
    rms       = 5.1e-3 # Jy/beam
    distance  = 140.0  # [pc]
    vsys      = 7.3    # systemic velocity [km/s]
    thr       = 5.


    ### main
    # for red
    outname = 'l1489_red'
    offbgn  = 200
    offend  = 400
    velbgn  = 51
    velend  = 68

    pvfit(imagename, outname ,rms, thr=thr, offbgn=offbgn, offend=offend, velbgn=velbgn, velend=velend, figsave=False, vsys=vsys)


    # for blue
    outname = 'l1489_blue'
    offbgn  = 200
    offend  = 400
    velbgn  = 3
    velend  = 20

    pvfit(imagename, outname ,rms, thr=thr, offbgn=offbgn, offend=offend, velbgn=velbgn, velend=velend, figsave=False, vsys=vsys)