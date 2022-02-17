import numpy as np
import matplotlib.pyplot as plt
from pvfit import PVFit



def main():
	# ------- input -------
	fitsfile= 'testfits/l1489.c18o.contsub.gain01.rbp05.mlt100.cf15.pbcor.pv69d.fits'

	outname = 'test'
	rms     = 5.1e-3 # Jy/beam
	thr     = 6.     # 6 sigma
	vsys    = 7.3
	incl    = 73.
	dist    = 140.
	clevels = np.array([-3,3,6,9,12,15,20,25,30])*rms
	xlim    = [-5, 5] # arcsec
	vlim    = [2, 12.5]
	Mlim    = [1, 2]
	# ---------------------


	# -------- main --------
	impv = PVFit(fitsfile, rms, vsys, dist, pa=69.)
	impv.get_edgeridge(outname, thr=thr, xlim=xlim, vlim=vlim, Mlim=Mlim, mode='gauss', incl=incl)
	impv.plotresults_pvdiagram(clevels=clevels)
	impv.plotresults_rvplane()
	#print(impv.results_sorted)

	#print(impv.results)
	#impv.pvfit_xcut(outname, rms, thr)
	#impv.plotresults_onpvdiagram(outname='test_xcut', clevels=clevels, colors=['r'])

	#impv = PVFit(fitsfile) # just to reflesh results recorded above
	#impv.pvfit_vcut(outname, rms, thr)
	#impv.plotresults_onpvdiagram(outname='test_vcut', clevels=clevels, colors=['r'])
	# ----------------------



if __name__ == '__main__':
	main()