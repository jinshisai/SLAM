
# -*- coding: utf-8 -*-

'''
Program to perform Gaussian fitting to a PV diagram.
Made and developed by J. Sai.

E-mail: jn.insa.sai@gmail.com

Latest update: 1/12/2020
'''


# modules
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits


# Constants (in cgs)
clight     = 2.99792458e10 # light speed [cm s^-1]



# Class
class Impvfits:
	'''
	Perform fitting to a PV diagram
	'''

	def __init__(self, infile):
		self.file = infile
		self.data, self.header = fits.getdata(infile, header=True)

		self.read_pvfits()

		self.results = []


	# Read fits file of Poistion-velocity (PV) diagram
	def read_pvfits(self):
		'''
		Read fits file of pv diagram produced by CASA.
		'''
		# read header
		header = self.header

		# number of axis
		naxis    = header['NAXIS']
		if naxis < 2:
			print ('ERROR\treadfits: NAXIS of fits is < 2.')
			return
		self.naxis = naxis

		naxis_i  = np.array([int(header['NAXIS'+str(i+1)]) for i in range(naxis)])
		label_i  = np.array([header['CTYPE'+str(i+1)] for i in range(naxis)])
		refpix_i = np.array([int(header['CRPIX'+str(i+1)]) for i in range(naxis)])
		refval_i = np.array([header['CRVAL'+str(i+1)] for i in range(naxis)]) # degree
		if 'CDELT1' in header:
			del_i = np.array([header['CDELT'+str(i+1)] for i in range(naxis)]) # degree
		self.naxis_i  = naxis_i
		self.label_i  = label_i
		self.refpix_i = refpix_i
		self.refval_i = refval_i

		# beam size (degree)
		if 'BMAJ' in header:
			bmaj     = header['BMAJ'] # degree
			bmin     = header['BMIN'] # degree
			bpa      = header['BPA']  # degree
			self.beam = np.array([bmaj*3600., bmin*3600., bpa]) # arcsec, arcsec, deg
		else:
			self.beam = None


		# Info. of P.A.
		if 'PA' in header:
			self.pa = header['PA']
		elif 'P.A.' in header:
			self.pa = header['P.A.']
		else:
			self.pa = None

		# Resolution along offset axis
		if self.pa:
			# an ellipse of the beam
			# (x/bmin)**2 + (y/bmaj)**2 = 1
			# y = x*tan(theta)
			# --> solve to get resolution in the direction of pv cut with P.A.=pa
			bmaj, bmin, bpa = self.beam
			del_pa = pa - bpa
			del_pa = del_pa*np.pi/180. # radian
			term_sin = (np.sin(del_pa)/bmin)**2.
			term_cos = (np.cos(del_pa)/bmaj)**2.
			res_off  = np.sqrt(1./(term_sin + term_cos))
			self.res_off = res_off
		else:
			self.res_off = None


		# rest frequency (Hz)
		if 'RESTFRQ' in header:
			restfreq = header['RESTFRQ']
		elif 'RESTFREQ' in header:
			restfreq = header['RESTFREQ']
		elif 'FREQ' in header:
			restfreq = header['FREQ']
		else:
			restfreq = None
		self.restfreq = restfreq


		# get axes
		# rotation of pixel coordinates
		if 'PC1_1' in header:
			pc_ij = np.array([
				[header['PC%i_%i'%(i+1,j+1)]
				if 'PC%i_%i'%(i+1,j+1) in header else 0.
				for j in range(naxis)] for i in range(naxis)])
			pc_ij = pc_ij*np.array([del_i[i] for i in range(naxis)])
		elif 'CD1_1' in header:
			pc_ij = np.array([
			[header['CD%i_%i'%(i+1,j+1)]
			if 'CD%i_%i'%(i+1,j+1) in header else 0.
			for j in range(naxis)] for i in range(naxis)])
		else:
			print ('CAUTION\tchannelmap: No keyword PCi_j or CDi_j are found. No rotation is assumed.')
			pc_ij = np.array([
				[1. if i==j else 0. for j in range(naxis)]
				 for i in range(naxis)])
			pc_ij = pc_ij*np.array([del_i[i] for i in range(naxis)])

		# axes
		axes = np.array([np.dot(pc_ij, (i+1 - refpix_i))\
		 for i in range(np.max(naxis_i))]).T # +1 in i+1 comes from 0 start index in python

		# x & v axes
		xaxis = axes[0]
		vaxis = axes[1]
		xaxis = xaxis[:naxis_i[0]]               # offset
		vaxis = vaxis[:naxis_i[1]] + refval_i[1] # frequency, absolute

		# check unit of offest
		if 'CUNIT1' in header:
			unit_i = np.array([header['CUNIT'+str(i+1)] for i in range(naxis)]) # degree
			if unit_i[0] == 'degree' or unit_i[0] == 'deg':
				# degree --> arcsec
				xaxis    = xaxis*3600.
				del_i[0] = del_i[0]*3600.
		else:
			print ('WARNING: No unit information in the header.\
				Assume units of arcesc and Hz for offset and frequency axes, respectively.')

		# frequency --> velocity
		if label_i[1] == 'VRAD' or label_i[1] == 'VELO':
			vaxis    = vaxis*1.e-3 # m/s --> km/s
			#del_v    = del_v*1.e-3
			#refval_v = refval_v*1.e-3
		else:
			print ('Convert frequency to velocity')
			vaxis    = clight*(1.-vaxis/restfreq) # radio velocity c*(1-f/f0) [cm/s]
			vaxis    = vaxis*1.e-5                # cm/s --> km/s
			#del_i[1] = -del_i[1]*clight/restfreq  # delf --> delv [cm/s]
			#del_i[1] = del_i[1]*1.e-5             # cm/s --> km/s

		if naxis == 2:
			axes_out = np.array([xaxis, vaxis], dtype=object)
		elif naxis == 3:
			saxis = axes[2]
			saxis = saxis[:naxis_i[2]]
			axes_out = np.array([xaxis, vaxis, saxis], dtype=object)
		else:
			print ('Error\tread_pvfits: naxis must be <= 3.')


		# get delta
		delx = xaxis[1] - xaxis[0]
		delv = vaxis[1] - vaxis[0]

		self.axes  = axes_out
		self.xaxis = xaxis
		self.vaxis = vaxis
		self.delx  = delx
		self.delv  = delv


	# Draw pv diagram
	def draw_pvdiagram(self,outname,data=None,header=None,ax=None,outformat='pdf',color=True,cmap='Greys',
		vmin=None,vmax=None,vsys=0,contour=True,clevels=None,ccolor='k', pa=None,
		vrel=False,logscale=False,x_offset=False,ratio=1.2, prop_vkep=None,fontsize=14,
		lw=1,clip=None,plot_res=True,inmode='fits',xranges=[], yranges=[],
		ln_hor=True, ln_var=True, alpha=None):
		'''
		Draw a PV diagram.

		Args:
		 - outname:
		'''

		# Modules
		import copy
		import matplotlib as mpl

		# format
		formatlist = np.array(['eps','pdf','png','jpeg'])

		# properties of plots
		#mpl.use('Agg')
		plt.rcParams['font.family']     ='Arial' # font (Times New Roman, Helvetica, Arial)
		plt.rcParams['xtick.direction'] = 'in'   # directions of x ticks ('in'), ('out') or ('inout')
		plt.rcParams['ytick.direction'] = 'in'   # directions of y ticks ('in'), ('out') or ('inout')
		plt.rcParams['font.size']       = fontsize  # fontsize

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


		# output file
		if (outformat == formatlist).any():
			outname = outname + '.' + outformat
		else:
			print ('ERROR\tsingleim_to_fig: Outformat is wrong.')
			return

		# Input
		if inmode == 'data':
			if data is None:
				print ("inmode ='data' is selected. data must be provided.")
				return
			naxis = len(data.shape)
		else:
			data   = self.data
			header = self.header
			naxis  = self.naxis


		# figures
		if ax:
			pass
		else:
			fig = plt.figure(figsize=(11.69,8.27)) # figsize=(11.69,8.27)
			ax  = fig.add_subplot(111)

		# Read
		xaxis = self.xaxis
		vaxis = self.vaxis
		delx  = self.delx
		delv  = self.delv
		nx    = len(xaxis)
		nv    = len(vaxis)

		# Beam
		bmaj, bmin, bpa = self.beam

		if self.res_off:
			res_off = self.res_off
		else:
			# Resolution along offset axis
			if self.pa:
				pa = self.pa

			if pa:
				# an ellipse of the beam
				# (x/bmin)**2 + (y/bmaj)**2 = 1
				# y = x*tan(theta)
				# --> solve to get resolution in the direction of pv cut with P.A.=pa
				del_pa = pa - bpa
				del_pa = del_pa*np.pi/180. # radian
				term_sin = (np.sin(del_pa)/bmin)**2.
				term_cos = (np.cos(del_pa)/bmaj)**2.
				res_off  = np.sqrt(1./(term_sin + term_cos))
			else:
				res_off = bmaj

		# relative velocity or LSRK
		offlabel = r'$\mathrm{Offset\ (arcsec)}$'
		if vrel:
			vaxis   = vaxis - vsys
			vlabel  = r'$\mathrm{Relative\ velocity\ (km\ s^{-1})}$'
			vcenter = 0
		else:
			vlabel  = r'$\mathrm{LSR\ velocity\ (km\ s^{-1})}$'
			vcenter = vsys


		# set extent of an image
		offmin = xaxis[0] - delx*0.5
		offmax = xaxis[-1] + delx*0.5
		velmin = vaxis[0] - delv*0.5
		velmax = vaxis[-1] + delv*0.5


		# set axes
		if x_offset:
			data   = data[0,:,:]
			extent = (offmin,offmax,velmin,velmax)
			xlabel = offlabel
			ylabel = vlabel
			hline_params = [vsys,offmin,offmax]
			vline_params = [0.,velmin,velmax]
			res_x = res_off
			res_y = delv
		else:
			data   = np.rot90(data[0,:,:])
			extent = (velmin,velmax,offmin,offmax)
			xlabel = vlabel
			ylabel = offlabel
			hline_params = [0.,velmin,velmax]
			vline_params = [vcenter,offmin,offmax]
			res_x = delv
			res_y = res_off


		# set colorscale
		if vmax:
			pass
		else:
			vmax = np.nanmax(data)


		# logscale
		if logscale:
			norm = mpl.colors.LogNorm(vmin=vmin,vmax=vmax)
		else:
			norm = mpl.colors.Normalize(vmin=vmin,vmax=vmax)


		# clip data at some value
		data_color = copy.copy(data)
		if clip:
			data_color[np.where(data < clip)] = np.nan

		# plot images
		if color:
			imcolor = ax.imshow(data_color, cmap=cmap, origin='lower',
				extent=extent, norm=norm, alpha=alpha)

		if contour:
			imcont  = ax.contour(data, colors=ccolor, origin='lower',
				extent=extent, levels=clevels, linewidths=lw, alpha=alpha)


		# axis labels
		ax.set_xlabel(xlabel)
		ax.set_ylabel(ylabel)

		# set xlim, ylim
		if len(xranges) == 0:
			ax.set_xlim(extent[0],extent[1])
		elif len(xranges) == 2:
			xmin, xmax = xranges
			ax.set_xlim(xmin, xmax)
		else:
			print ('WARRING: Input xranges is wrong. Must be [xmin, xmax].')
			ax.set_xlim(extent[0],extent[1])

		if len(yranges) == 0:
			ax.set_ylim(extent[2],extent[3])
		elif len(yranges) == 2:
			ymin, ymax = yranges
			ax.set_ylim(ymin, ymax)
		else:
			print ('WARRING: Input yranges is wrong. Must be [ymin, ymax].')
			ax.set_ylim(extent[2],extent[3])


		# lines showing offset 0 and relative velocity 0
		if ln_hor:
			xline = plt.hlines(hline_params[0], hline_params[1], hline_params[2], ccolor, linestyles='dashed', linewidths = 1.)
		if ln_var:
			yline = plt.vlines(vline_params[0], vline_params[1], vline_params[2], ccolor, linestyles='dashed', linewidths = 1.)

		ax.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True, pad=9)

		# plot resolutions
		if plot_res:
			# x axis
			#print (res_x, res_y)
			res_x_plt, res_y_plt = ax.transLimits.transform((res_x*0.5, res_y*0.5)) -  ax.transLimits.transform((0, 0)) # data --> Axes coordinate
			ax.errorbar(0.1, 0.1, xerr=res_x_plt, yerr=res_y_plt, color=ccolor, capsize=3, capthick=1., elinewidth=1., transform=ax.transAxes)

		# aspect ratio
		if ratio:
			change_aspect_ratio(ax, ratio)

		# save figure
		plt.savefig(outname, transparent=True)

		return ax


class PVFit(Impvfits):

	'''
	Perform fitting to a PV diagram
	'''

	# Initializing
	def __init__(self, infile):
		super().__init__(infile)

		self.outname = 'object'
		self.results = {}
		self.dicindx = 0


	def read_results(self, infiles):
		# initialize
		self.results = {}
		self.dicindx = 0

		if type(infiles) != list:
			print ('ERROR\tread_results: infiles must be the list type.')
			return

		for i in range(len(infiles)):
			self.results.update({'%i'%i: np.genfromtxt(infiles[i], unpack=Trued, dtype='float64')})
			self.dicindx += 1


	# pvfit
	def pvfit_vcut(self, outname, rms, thr, xlim=[], vlim=[], pixrng = 2):
		'''
		Fitting along the velocity axis.

		Args:
		 - outname
		'''
		# modules
		import math
		from scipy import optimize
		from mpl_toolkits.axes_grid1 import ImageGrid


		# functions
		# gaussian
		def gaussian(x,amp,mean,sig):
			#coeff=A/(np.sqrt(2.0*np.pi)*sig)
			exp=np.exp(-(x-mean)*(x-mean)/(2.0*sig*sig))
			gauss=amp*exp
			return gauss

		#def fit_error()
		# y-f(x)/yerr, contents of  xi square
		def chi_gauss(param, xdata, ydata, ysig):
			chi = (ydata - (gaussian(xdata, *param))) / ysig
			return chi



		# remember output name
		self.outname = outname


		# data
		data = self.data

		if self.naxis == 2:
			pass
		elif self.naxis == 3:
			data = data[0,:,:] # Remove stokes I
		else:
			print ('Error\tpvfit_vcut: n_axis must be 2 or 3.')
			return


		# axes
		xaxis = self.xaxis
		vaxis = self.vaxis


		# resolution
		if self.res_off:
			res_off = self.res_off
			hob     = np.round((res_off*0.5/self.delx), 0) # harf of beamsize (pix)
		else:
			bmaj, bmin, bpa = self.beam
			res_off = (bmaj + bmin)*0.5                # in arcsec
			hob     = np.round((res_off*0.5/self.delx), 0) # harf of beamsize [pix]

		# sampling step; half of beam
		hob = int(hob) # (pix)


		# relative velocity or LSRK
		xlabel = 'Offset (arcsec)'
		vlabel = r'LSR velocity ($\mathrm{km~s^{-1}}$)'


		# list to save final results
		res_v = np.array([], dtype='float64')
		res_x = np.array([], dtype='float64')
		err_x = np.array([], dtype='float64')
		err_v = np.array([], dtype='float64')
		res_chi2 = np.array([], dtype='float64')


		### calculate intensity-weighted mean positions
		# cutting at an velocity and determining the position

		# x & v ranges used for calculation

		# for fitting
		xaxis_fit = xaxis
		vaxis_fit = vaxis
		data_fit  = data

		if len(xlim) == 2:
			xaxis_fit = xaxis_fit[np.where((xaxis >= xlim[0]) & (xaxis <= xlim[-1]))]
			data_fit  = data_fit[:,np.where((xaxis >= xlim[0]) & (xaxis <= xlim[-1]))[0]]
			print ('x range: %.2f--%.2f arcsec'%(np.nanmin(xaxis_fit), np.nanmax(xaxis_fit)))
			#print (data_fit.shape)
		elif len(xlim) == 0:
			pass
		else:
			print ('Warning\tpvfit_vcut: Size of xlim is not correct.\
			 xlim must be given as [xmin, xmax]. Given xlim is ignored.')

		if len(vlim) == 2:
			vaxis_fit = vaxis_fit[np.where((vaxis >= vlim[0]) & (vaxis <= vlim[-1]))]
			data_fit  = data_fit[np.where((vaxis >= vlim[0]) & (vaxis <= vlim[-1]))[0],:]
			print ('v range: %.2f--%.2f km/s'%(np.nanmin(vaxis_fit), np.nanmax(vaxis_fit)))
		elif len(vlim) == 0:
			pass
		else:
			print ('Warning\tpvfit_vcut: Size of vlim is not correct.\
			 vlim must be given as [vmin, vmax]. Given vlim is ignored.')


		# Nyquist sampling
		nx_fit   = len(xaxis_fit)
		loop_max = nx_fit//hob +1

		#print(hob)
		npanel = loop_max
		ncol = math.ceil(np.sqrt(npanel))
		ncol = int(ncol)
		nrow = 1
		while ncol*nrow <= npanel:
			nrow += 1


		# figure for check result
		fig  = plt.figure(figsize=(11.69, 8.27))
		grid = ImageGrid(fig, rect=111, nrows_ncols=(nrow,ncol),
		axes_pad=0,share_all=True, aspect=False, label_mode='L') #,cbar_mode=cbar_mode)
		gridi = 0

		# x & y label
		grid[(nrow*ncol-ncol)].set_xlabel(r'Velocity ($\mathrm{km\ s^{-1}}$)')
		grid[(nrow*ncol-ncol)].set_ylabel(r'Intensity ($\mathrm{Jy\ beam^{-1}}$)')



		# calculation
		ii = 0 # pix
		for i in range(loop_max):
			# where it is now!!!
			# Nyquist sampling
			ii = i*hob
			#print(ii, i*hob)

			if ii >= nx_fit:
				continue

			xi = xaxis_fit[ii]
			print ('x_now: %.2f arcsec'%xi)

			d_i = data_fit[:, ii]
			snr = np.nanmax(d_i)/rms
			posacc = res_off/snr

			# Enough high snr?
			if thr:
				if snr < thr:
					continue


			# determining used data range
			if pixrng:
				# error check
				if type(pixrng) != int:
					print ('Error\tpvfit_vcut: pixrng must be integer.')
					return

				# use pixels only around intensity peak
				peakindex = np.argmax(d_i)
				vfit_indx = [peakindex - pixrng, peakindex + pixrng+1]

				d_i  = d_i[vfit_indx[0]:vfit_indx[1]]
				v_i  = vaxis_fit[vfit_indx[0]:vfit_indx[1]]
				nd_i = len(d_i)

			# skip if pixel number for calculation is below 3
			# just for safety
			if d_i.size <= 3:
				gridi = gridi + 1
				continue


			# set the initial parameter
			sigx = 0.5
			mx   = v_i[np.argmax(d_i)]
			amp  = np.nanmax(d_i)*np.sqrt(2.*np.pi)*sigx
			pinp = [amp, mx, sigx]


			# fitting
			results = optimize.leastsq(chi_gauss, pinp, args=(v_i, d_i, rms), full_output=True)

			# recording output results
			# param_out: fitted parameters
			# err: error of fitted parameters
			# chi2: chi-square
			# DOF: degree of freedum
			# reduced_chi2: reduced chi-square
			param_out    = results[0]
			param_cov    = results[1]
			chi2         = np.sum(chi_gauss(param_out, v_i, d_i, rms)**2.)
			ndata        = len(d_i)
			nparam       = len(param_out)
			dof          = ndata - nparam - 1
			reduced_chi2 = chi2/dof

			#print (ndata, nparam, dof)
			#print (param_cov)
			if (dof >= 0) and (param_cov is not None):
				param_cov = param_cov*reduced_chi2
			else:
				param_cov = np.full((nparam, nparam),np.inf)

			# best fit value
			amp_fit, mx_fit, sigx_fit = param_out

			# fitting error
			param_err = np.array([
				np.abs(param_cov[j][j])**0.5 for j in range(nparam)
				])
			amp_fit_err, mx_fit_err, sigx_fit_err = param_err



			res_x = np.append(res_x, xi)
			res_v = np.append(res_v, mx_fit)
			err_x = np.append(err_x, posacc)
			err_v = np.append(err_v, mx_fit_err)
			res_chi2 = np.append(res_chi2, reduced_chi2)
			#print 'offmean: ', offmean, ' arcsec'
			#print 'mean error: ', offm_err, ' arcsec'


			# print results
			print ('Chi2: ', reduced_chi2)
			print ('Fitting results: amp   mean   sigma')
			print (amp_fit, mx_fit, sigx_fit)
			print ('Errors')
			print (amp_fit_err, mx_fit_err, sigx_fit_err)


			# plot results
			ax = grid[gridi]

			# observed data
			ax.step(vaxis_fit, data_fit[:,ii], linewidth=2., color='grey', where='mid')
			ax.step(v_i, d_i, linewidth=3., color='k', where='mid')
			#ax.errorbar(velcalc, Icalc, yerr=rms, fmt= 'k.', color = 'k', capsize = 2, capthick = 1)

			# fitted Gaussian
			x_model = np.linspace(np.nanmin(vaxis_fit), np.nanmax(vaxis_fit), 256) # offset axis for plot
			g_model = gaussian(x_model, amp_fit, mx_fit, sigx_fit)
			ax.plot(x_model, g_model, lw=2., color='r', ls='-', alpha=0.8)

			# offset label
			ax.text(0.9, 0.9, '%03.2f'%xi, horizontalalignment='right',
			    verticalalignment='top',transform=ax.transAxes)

			ax.tick_params(which='both', direction='in',bottom=True, top=True, left=True, right=True, pad=9)

			# step
			gridi = gridi + 1


		# Store
		self.results.update({'%i'%self.dicindx:np.array([res_x, res_v, err_x, err_v, res_chi2], dtype=object)})
		self.dicindx += 1

		#plt.show()
		fig.savefig(outname+"_chi2_pv_vfit.pdf", transparent=True)
		plt.clf()


		### output results as .txt file
		res_hd  = 'offset[arcsec]\tvelocity[km/s]\toff_err[arcesc]\tvel_err[km/s]'
		res_all = np.c_[res_x, res_v, err_x, err_v]
		np.savetxt(outname+'_chi2_pv_vfit.txt', res_all,fmt = '%4.4f', delimiter ='\t', header = res_hd)


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


		if inner_threshold:
			indx     = np.where(np.abs(offset) >= inner_threshold)
			offset   = offset[indx]
			velocity = velocity[indx]
			off_err  = off_err[thrindx]
			vel_err  = vel_err[thrindx]

		if outer_threshold:
			indx     = np.where(np.abs(offset) >= outer_threshold)
			offset   = offset[indx]
			velocity = velocity[indx]
			off_err  = off_err[thrindx]
			vel_err  = vel_err[thrindx]

		if mode == 'sp':
			func = chi_splaw
		elif mode == 'dp':
			func     = chi_dplaw
			velocity = np.abs(velocity - vsys)
		else:
			print ('Error: mode must be sp or dp.')
			return

		# fitting
		ndata        = len(offset)
		nparams      = len(params_inp)
		result       = optimize.leastsq(func, params_inp, args=(offset, velocity, off_err, vel_err), full_output = True)
		params_out   = result[0]
		chi          = func(params_out, offset, velocity, off_err, vel_err)
		chi2         = np.sum(chi*chi)
		dof          = ndata - nparams -1
		reduced_chi2 = chi2/dof

		# output
		print ('Results:')
		#print 'chi2', chi2
		print ('')
		with np.printoptions(precision=4, suppress=True):
			print (params_out)
		print ('reduced chi^2: %.2f'%reduced_chi2)
		#print ('degree of freedum: %.f'%DOF)

		# error
		perrors = estimate_perror(params_out, func, offset, velocity, off_err, vel_err, niter=3000)

		self.params_out = params_out
		self.perrors    = perrors



	# Plot results
	def plotresults_onpvdiagram(self, marker='o', color='r', alpha=1.,
		data=None,header=None,ax=None,outformat='pdf',pvcolor=True,cmap='Greys',
		vmin=None,vmax=None,vsys=0,contour=True,clevels=None,pvccolor='k', pa=None,
		vrel=False,logscale=False,x_offset=False,ratio=1.2, prop_vkep=None,fontsize=14,
		lw=1,clip=None,plot_res=True,inmode='fits',xranges=[], yranges=[],
		ln_hor=True, ln_var=True, pvalpha=None):
		'''

		'''
		outname = self.outname + '_chi2_pv_vfit_pvdiagram'

		ax = self.draw_pvdiagram(outname, data=data, header=header, ax=ax,
			outformat=outformat, color=pvcolor, cmap=cmap,
			vmin=vmin, vmax=vmax, vsys=vsys, contour=contour, clevels=clevels,
			ccolor=pvccolor, pa=pa, vrel=vrel, logscale=logscale, x_offset=x_offset,
			ratio=ratio, prop_vkep=prop_vkep, fontsize=fontsize, lw=lw, clip=clip,
			plot_res=plot_res, inmode=inmode, xranges=xranges, yranges=yranges,
			ln_hor=ln_hor, ln_var=ln_var, alpha=pvalpha)


		res_x, res_v, err_x, err_v, res_chi2 = self.results

		if x_offset:
			ax.errorbar(res_x, res_v, xerr=err_x, yerr=err_v,
				color=color, marker=marker, alpha=alpha, capsize=2.,
				capthick=2., linestyle='')
		else:
			ax.errorbar(res_v, -res_x, xerr=err_v, yerr=err_x,
				color=color, marker=marker, alpha=alpha, capsize=2.,
				capthick=2., linestyle='')

		plt.savefig(outname+'.'+outformat, transparent=True)

		return ax


	def plotresults_onrvplane(self, outformat='pdf', ax=None, vsys=0., xlog=True, ylog=True,
		marker='o', color='r', xlim=[], ylim=[]):
		'''
		'''
		res_x, res_v, err_x, err_v, res_chi2 = self.results

		outname = self.outname + '_chi2_pv_vfit_rvplane'

		if ax:
			pass
		else:
			fig = plt.figure(figsize=(8.27, 8.27))
			ax  = fig.add_subplot(111)

		ax.errorbar(np.abs(res_x), np.abs(res_v -vsys), xerr=err_x, yerr=err_v,
			marker=marker, color=color, capsize=2., capthick=2., linestyle='')

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

		plt.savefig(outname+'.'+outformat, transparent=True)

		return ax
