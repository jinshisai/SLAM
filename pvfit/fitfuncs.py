'''
Fitting functions
'''

# modules
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import optimize
from scipy.stats import norm, uniform





# read file
def read_pvfitres(fname, inner_threshold=None, outer_threshold=None, toau=True, dist=140.):
	'''
	Read fitting-result files.
	'''
	# read files
	offset, velocity, offerr, velerr = np.genfromtxt(fname, comments='#', unpack = True)

	# offset threshold of used data point
	if inner_threshold:
		thrindx  = np.where(np.abs(offset) >= inner_threshold)
		offset   = offset[thrindx]
		velocity = velocity[thrindx]
		offerr   = offerr[thrindx]
		velerr   = velerr[thrindx]

	# offset threshold of used data point
	if outer_threshold:
		thrindx  = np.where(np.abs(offset) <= outer_threshold)
		offset   = offset[thrindx]
		velocity = velocity[thrindx]
		offerr   = offerr[thrindx]
		velerr   = velerr[thrindx]

	if toau:
		offset = offset*dist
		offerr = offerr*dist

	return offset, velocity, offerr, velerr




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
	if nparams == 3:
		xlabels    = [r'$V_\mathrm{sys}$', r'$V_\mathrm{100}$', r'$p$' ]
	elif nparams == 4:
		xlabels    = [r'$V_\mathrm{break}$', r'$R_\mathrm{break}$', r'$p_\mathrm{in}$', r'$p_\mathrm{out}$']
	else:
		xlabels = [r'$p%i$'%(i+1) for i in range(nparams)]

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