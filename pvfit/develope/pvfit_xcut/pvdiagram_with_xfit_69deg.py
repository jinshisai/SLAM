# coding: utf-8

### making diagram


### import modules
import os
import numpy as np
import pyfigures as pyfg
import matplotlib.pyplot as plt
from astrconsts import *
from astropy.io import fits




### PV diagram
# Keplerian rotation velocity
def Kep_rot(r, M):
    # input parameter's unit
    # r: [au]
    # M: [g]

    # convering units
    r = r*auTOcm # au --> cm
    Vkep = np.sqrt(Ggrav*M/r) # cm/s
    Vkep = Vkep*1.e-5 # cm/s --> km/s
    return Vkep




if __name__ == '__main__':
	fitsdata = "../../../l1489.c18o.contsub.gain01.rbp05.mlt100.cf15.pbcor.pv69d.fits"
	rms      = 5.1e-3 # Jy/beam, for no taper
	vrel     = False
	vsys     = 7.3
	#vsys     = 7.22
	outname  = 'pvdiagram_l1489_c18o_notaper_69deg_xfit'
	outformat= 'pdf'
	clevels  = np.array([-9,-6,-3., 3., 5., 7., 9., 11., 13., 15., 20., 25.,30.])*rms
	color    = True
	contour  = True
	vmin     = 0
	vmax     = 16.*rms
	#vmax     = None
	clip     = 3.*rms
	#clip     = None
	ccolor   = 'k'
	cmap     = 'Greys'
	logscale = False
	x_offset = False
	Mstar    = 1.6*Msun   # Msun
	Mstar    = 1.26*Msun  # before the correction
	dist     = 140.       # pc
	inc      = 73.        # from 1.3 mm continuum
	offthr   = 100./140.  # offset threshold of used data points, from Aso et al. (2015)
	offthr   = None
	velthr   = None
	#print data.shape


	### plot figure
	ax = pyfg.pvdiagram(fitsdata,outname,outformat=outformat,color=True,cmap=cmap,
		vmin=vmin,vmax=vmax,contour=True,clevels=clevels,ccolor=ccolor,
		vrel=vrel,vsys=vsys,logscale=False,x_offset=False,fontsize=14, clip=clip)


	# Keplerian velocity
	r_au      = np.logspace(0,3,50)
	vrot      = Kep_rot(r_au, Mstar)*np.sin(np.radians(inc))
	vrot_red  = vrot + vsys
	vrot_blue = vsys - vrot


	# plot
	#ax.plot(vrot_red, -r_au/dist,linestyle='-',lw=2, color='limegreen')
	#ax.plot(vrot_blue, r_au/dist,linestyle='-',lw=2, color='limegreen')



	### representative positions
	# reading files
	fred  = 'l1489_red_pv_xfit.txt'
	fblue = 'l1489_blue_pv_xfit.txt'

	offset_red, velocity_red, offerr_red, velerr_red     = np.genfromtxt(fred, comments='#', unpack = True)
	offset_blue, velocity_blue, offerr_blue, velerr_blue = np.genfromtxt(fblue, comments='#', unpack = True)

	offset_red  = -offset_red
	offset_blue = -offset_blue

	if offthr:
		thrindx      = np.where(np.abs(offset_red) < offthr)
		offset_red   = offset_red[thrindx]
		velocity_red = velocity_red[thrindx]
		offerr_red   = offerr_red[thrindx]
		velerr_red   = velerr_red[thrindx]

		thrindx       = np.where(np.abs(offset_blue) < offthr)
		offset_blue   = offset_blue[thrindx]
		velocity_blue = velocity_blue[thrindx]
		offerr_blue   = offerr_blue[thrindx]
		velerr_blue   = velerr_blue[thrindx]

	if velthr:
		thrindx      = np.where(velocity_red - vsys < velthr)
		offset_red   = offset_red[thrindx]
		velocity_red = velocity_red[thrindx]
		offerr_red   = offerr_red[thrindx]
		velerr_red   = velerr_red[thrindx]

		thrindx       = np.where(vsys - velocity_blue < velthr)
		offset_blue   = offset_blue[thrindx]
		velocity_blue = velocity_blue[thrindx]
		offerr_blue   = offerr_blue[thrindx]
		velerr_blue   = velerr_blue[thrindx]


	# overploting fitting results
	#myblue = (0.2,0.2,1)
	ax.errorbar(velocity_red, offset_red, xerr = velerr_red, yerr = offerr_red, fmt= '.', color = 'red', capsize = 2, capthick = 1)
	ax.errorbar(velocity_blue, offset_blue, xerr = velerr_blue, yerr = offerr_blue, fmt= '.', color = 'blue', capsize = 2, capthick = 1)


	# setting axes
	data, header = fits.getdata(fitsdata, header=True)

	noff       = int(header['NAXIS1'])
	nvel       = int(header['NAXIS2'])
	bmaj       = header['BMAJ']*60.*60. # in arcsec
	bmin       = header['BMIN']*60.*60. # in arcsec
	bpa        = header['BPA']          # [deg]
	bsize      = (bmaj + bmin)*0.5

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

	offmin = refval_off + (1 - refpix_off)*del_off     # refpix is not in 0 start
	offmax = refval_off + (noff - refpix_off)*del_off  # delpix (n - 1)
	velmin = refval_vel + (1 - refpix_vel)*del_vel
	velmax = refval_vel + (nvel - refpix_vel)*del_vel

	ax.set_xlim(velmin,velmax)
	ax.set_ylim(offmin,offmax)

	outfile = outname + '.' + outformat
	plt.savefig(outfile, transparent=True)
	#plt.show()