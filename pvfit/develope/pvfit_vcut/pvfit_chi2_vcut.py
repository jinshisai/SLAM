
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



# -------------- functions for MCMC Gaussian fit ----------------
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
def pvfit(self, outname, rms, thr, offbgn=None, offend=None,
    velbgn=None, velend=None, figsave=False, vsys=None, pixrng = 2, nthr=None):

    ###  reading fits files
    data, header=fits.getdata(self,header=True)
    print ('fits data shape: ',data.shape)



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
    hob = int(hob)

    # check unit
    if offunit == 'degree' or offunit == 'deg':
        refval_off = refval_off*60.*60.
        del_off    = del_off*60.*60.


    # relative velocity or LSRK
    offlabel = 'offset (arcsec)'
    vellabel = 'LSRK velocity (km/s)'


    # cell center, not edge of the image
    offmin = refval_off + (1 - refpix_off)*del_off    # refpix is not in 0 start
    offmax = refval_off + (noff - refpix_off)*del_off # delpix (n - 1)
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




    ### calculate intensity-weighted mean positions
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

    # Nyquist sampling
    offend2 = int(offbgn + ((offend - offbgn + 1)/hob) +1)

    npanel = offend2 - offbgn
    nrow = math.ceil(np.sqrt(npanel))
    nrow = int(nrow)
    ncol = nrow


    # figure for check result
    fig  = plt.figure(figsize=(11.69, 8.27))
    grid = ImageGrid(fig, rect=111, nrows_ncols=(nrow,ncol),
        axes_pad=0,share_all=True, aspect=False, label_mode='L') #,cbar_mode=cbar_mode)
    gridi = 0

    # x & y label
    grid[(nrow*ncol-ncol)].set_xlabel(r'velocity ($\mathrm{km\ s^{-1}}$)')
    grid[(nrow*ncol-ncol)].set_ylabel(r'intensity ($\mathrm{Jy\ beam^{-1}}$)')



    # calculation
    for n_row in range(offbgn,offend2):
        # where it is now!!!
        # Nyquist sampling
        n_row2 = offbgn + (n_row - offbgn)*hob
        offnow = offaxis[n_row2]
        print ('offset: ', offnow, ' arcsec')

        # set offset range used
        Icalc     = data[0, velbgn:velend, n_row2]
        velcalc   = velaxis[velbgn:velend]
        snr       = np.nanmax(Icalc)/rms
        posacc    = bsize/snr

        # determining used data range
        if pixrng:
            # error check
            if type(pixrng) != int:
                print ('Error\tpvfittin: pixrng must be integer.')
                return

            # use pixels only around intensity peak
            peakindex = np.argmax(Icalc)
            #print peakindex
            velcalcmin = peakindex - pixrng
            velcalcmax = peakindex + pixrng

            Icalc     = Icalc[velcalcmin:velcalcmax+1]
            velcalc   = velcalc[velcalcmin:velcalcmax+1]
            ncalc     = len(Icalc)


        # using data points >= thr * sigma
        if thr:
            if np.nanmax(Icalc) < thr*rms:
                continue

        # skip if pixel number for calculation is below 3
        if Icalc.size <= 3 or velcalc.size <= 3:
            gridi = gridi + 1
            continue


        # set the initial parameter
        sigx = 0.5
        mx   = velcalc[np.argmax(Icalc)]
        amp  = np.nanmax(Icalc)*np.sqrt(2.*np.pi)*sigx
        pinp = [amp, mx, sigx]


        # fitting
        results = optimize.leastsq(chi, pinp, args=(velcalc, Icalc, rms), full_output=True)

        # recording output results
        # param_out: fitted parameters
        # err: error of fitted parameters
        # chi2: chi-square
        # DOF: degree of freedum
        # reduced_chi2: reduced chi-square
        param_out    = results[0]
        param_cov    = results[1]
        chi2         = np.sum(chi(param_out, velcalc, Icalc, rms)**2.)
        ndata        = len(velcalc)
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
        for j in range(nparam):
            try:
                param_err.append(np.abs(param_cov[j][j])**0.5)
            except:
                print ('Cannot calculate covariance. Parameter error will be set as 0.')
                param_err.append(0.00)
        amp_fit_err, mx_fit_err, sigx_fit_err = param_err



        res_off = np.append(res_off, offnow)
        res_vel = np.append(res_vel, mx_fit)
        err_off = np.append(err_off, posacc)
        err_vel = np.append(err_vel, mx_fit_err)
        #print 'offmean: ', offmean, ' arcsec'
        #print 'mean error: ', offm_err, ' arcsec'


        # print results
        print ('Chi2: ', reduced_chi2)
        print ('Fitting results: amp   mean   sigma')
        print (amp_fit, mx_fit, sigx_fit)
        print ('Errors')
        print (amp_fit_err, mx_fit_err, sigx_fit_err)


        ### plot results
        ax = grid[gridi]

        # observed data
        ax.errorbar(velcalc, Icalc, yerr=rms, fmt= 'k.', color = 'k', capsize = 2, capthick = 1)

        # fitted Gaussian
        off_forgauss = np.linspace(np.nanmin(velcalc), np.nanmax(velcalc), 256) # offset axis for plot
        gauss        = gaussian(off_forgauss, amp_fit, mx_fit, sigx_fit)
        ax.plot(off_forgauss, gauss, lw=1, color='r', ls='-')

        # velcoity label
        ax.text(0.9, 0.9, '%03.2f'%-offnow, horizontalalignment='right',
            verticalalignment='top',transform=ax.transAxes)



        # step
        gridi = gridi + 1


    #plt.show()
    fig.savefig(outname+"_chi2_pv_vfit.pdf", transparent=True)
    plt.clf()


    ### output results as .txt file
    res_hd  = '\t offset[arcsec]\t velocity[km/s]\t off_err[arcesc]\t vel_err[km/s]'
    res_all = np.c_[res_off, res_vel, err_off, err_vel]
    np.savetxt(outname+'_chi2_pv_vfit.txt', res_all,fmt = '%04.4f', delimiter ='\t', header = res_hd)



    if figsave:
        # set figure
        fig  = plt.figure(figsize=(11.69, 8.27))
        ax   = fig.add_subplot(111)

        # set axes
        redata  = np.rot90(data[0,:,:])
        res_off = -res_off
        extent  = (velmin - 0.5*del_vel, velmax + 0.5*del_vel, offmin - 0.5*del_off, offmax + 0.5*del_off )
        xlabel  = vellabel
        ylabel  = offlabel


        # plot images
        clevels = np.array([-3., 3., 5., 7., 9., 11., 13., 15., 20., 25., 30., 40., 50., 60., 70.])*rms
        imcont  = ax.contour(redata, colors='k', origin='lower',extent=extent, levels=clevels, linewidths=1)
        ax.errorbar(res_vel, res_off, xerr=err_vel, yerr=err_off, fmt='o', color='royalblue', capsize=1)

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
        fig.savefig(outname+'_chi2_pv.pdf', transparent=True)

        return ax

    return



if __name__ == '__main__':
    ### input parameters
    imagename = '/path/to/fits file'
    rms       = rms    # Jy/beam
    distance  = 140.0  # [pc]
    vsys      = 7.3    # systemic velocity [km/s]
    thr       = 10.
    pixrng    = 2
    nthr      = None

    ### main
    # for red
    outname = 'l1489_red'
    offbgn  = None
    offend  = None
    velbgn  = None
    velend  = None

    pvfit(imagename, outname ,rms, thr=thr, offbgn=offbgn, offend=offend, velbgn=velbgn, velend=velend, figsave=True, vsys=vsys, pixrng=pixrng, nthr=nthr)


    # for blue
    outname = 'l1489_blue'
    offbgn  = None
    offend  = None
    velbgn  = None
    velend  = None

    pvfit(imagename, outname ,rms, thr=thr, offbgn=offbgn, offend=offend, velbgn=velbgn, velend=velend, figsave=True, vsys=vsys, pixrng=pixrng, nthr =nthr)
