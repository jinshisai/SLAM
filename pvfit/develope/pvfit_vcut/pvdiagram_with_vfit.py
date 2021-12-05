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
	### input parameters
	fitsdata        = '/path/to/fits file'
	fred            = 'xxx_red_chi2_pv_vfit.txt'
	fblue           = 'xxx_blue_chi2_pv_vfit.txt'
	outname         = 'outputname'
	rms             = 5.4e-3 # Jy/beam, for taper 200k
	vrel            = False
	vsys            = 7.3
	outformat       = 'pdf'
	clevels         = np.array([-3., 3., 6., 12., 24., 36., 48., 60.])*rms
	#clevels         = np.array([-3., 3., 6., 12., 24., 36., 48., 60.])*rms
	color           = True
	contour         = True
	vmin            = 0
	vmax            = 40.*rms
	clip            = 3.*rms
	ccolor          = 'k'
	cmap            = 'Greys'
	logscale        = False
	x_offset        = False
	Mstar           = 1.6*Msun  # Msun
	dist            = 140.       # pc
	inc             = 73.        # from 1.3 mm continuum
	offthr_red_in   = 2.
	offthr_red_out  = None
	offthr_blue_in  = 2.
	offthr_blue_out = None
	velthr          = None
	#print data.shape


	### plot figure
	ax = pyfg.pvdiagram(fitsdata,outname,outformat=outformat,color=True,cmap=cmap,
		vmin=vmin,vmax=vmax,contour=True,clevels=clevels,ccolor=ccolor,
		vrel=vrel,vsys=vsys,logscale=False,x_offset=False,fontsize=14, clip=clip)


	'''
	# Keplerian velocity
	# inc 60 deg
	inc = 60.
	r_au      = np.logspace(0,3,50)
	vrot      = Kep_rot(r_au, Mstar)*np.sin(np.radians(inc))
	vrot_red  = vrot + vsys
	vrot_blue = vsys - vrot
	ax.plot(vrot_red, -r_au/dist,linestyle='-',lw=2, color='orange')
	ax.plot(vrot_blue, r_au/dist,linestyle='-',lw=2, color='orange',label=r'$i=60\degree$')

	# inc 73 deg
	inc = 73.
	r_au      = np.logspace(0,3,50)
	vrot      = Kep_rot(r_au, Mstar)*np.sin(np.radians(inc))
	vrot_red  = vrot + vsys
	vrot_blue = vsys - vrot
	ax.plot(vrot_red, -r_au/dist,linestyle='-',lw=2, color='red')
	ax.plot(vrot_blue, r_au/dist,linestyle='-',lw=2, color='red',label=r'$i=73\degree$')

	# inc 80 deg
	inc = 80.
	r_au      = np.logspace(0,3,50)
	vrot      = Kep_rot(r_au, Mstar)*np.sin(np.radians(inc))
	vrot_red  = vrot + vsys
	vrot_blue = vsys - vrot
	#ax.plot(vrot_red, -r_au/dist,linestyle='-',lw=2, color='lightgreen')
	#ax.plot(vrot_blue, r_au/dist,linestyle='-',lw=2, color='lightgreen',label=r'$i=80\degree$')

	ax.legend(edgecolor='white', loc='upper right')
	'''



	### representative positions
	# reading files
	offset_red, velocity_red, offerr_red, velerr_red     = np.genfromtxt(fred, comments='#', unpack = True)
	offset_blue, velocity_blue, offerr_blue, velerr_blue = np.genfromtxt(fblue, comments='#', unpack = True)

	offset_red  = -offset_red
	offset_blue = -offset_blue

	if offthr_red_in:
		thrindx      = np.where(np.abs(offset_red) >= offthr_red_in)
		offset_red   = offset_red[thrindx]
		velocity_red = velocity_red[thrindx]
		offerr_red   = offerr_red[thrindx]
		velerr_red   = velerr_red[thrindx]

	if offthr_blue_in:
		thrindx       = np.where(np.abs(offset_blue) >= offthr_blue_in)
		offset_blue   = offset_blue[thrindx]
		velocity_blue = velocity_blue[thrindx]
		offerr_blue   = offerr_blue[thrindx]
		velerr_blue   = velerr_blue[thrindx]

	if offthr_red_out:
		thrindx      = np.where(np.abs(offset_red) <= offthr_red_out)
		offset_red   = offset_red[thrindx]
		velocity_red = velocity_red[thrindx]
		offerr_red   = offerr_red[thrindx]
		velerr_red   = velerr_red[thrindx]

	if offthr_blue_out:
		thrindx       = np.where(np.abs(offset_blue) <= offthr_blue_out)
		offset_blue   = offset_blue[thrindx]
		velocity_blue = velocity_blue[thrindx]
		offerr_blue   = offerr_blue[thrindx]
		velerr_blue   = velerr_blue[thrindx]

	if velthr:
		thrindx      = np.where(velocity_red - vsys > velthr)
		offset_red   = offset_red[thrindx]
		velocity_red = velocity_red[thrindx]
		offerr_red   = offerr_red[thrindx]
		velerr_red   = velerr_red[thrindx]

		thrindx       = np.where(vsys - velocity_blue > velthr)
		offset_blue   = offset_blue[thrindx]
		velocity_blue = velocity_blue[thrindx]
		offerr_blue   = offerr_blue[thrindx]
		velerr_blue   = velerr_blue[thrindx]


	# overploting fitting results
	ax.errorbar(velocity_red, offset_red, xerr = velerr_red, yerr = offerr_red, fmt= 'r.', color = 'red', capsize = 2, capthick = 1)
	ax.errorbar(velocity_blue, offset_blue, xerr = velerr_blue, yerr = offerr_blue, fmt= 'b.', color = 'blue', capsize = 2, capthick = 1)


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