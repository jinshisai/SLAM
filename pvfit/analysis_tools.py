'''
Analysis tools for PVfit

'''

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt


# read file
def read_pvfitres(fname, inner_threshold=None, outer_threshold=None, toau=True, dist=140.):
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