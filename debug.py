import numpy as np
import matplotlib.pyplot as plt
from pvfit import PVFit



def main():
    # ------- input -------
    #fitsfile = 'testfits/l1489.c18o.contsub.gain01.rbp05.mlt100.cf15.pbcor.pv69d.fits'
    #vsys    = 7.3
    #incl    = 73.
    #pa      = 69.
    #fitsfile = 'testfits/tmc1a.c18o.contsub.rbp05.mlt100.clean.majpv.fits'
    #vsys    = 6.4
    #incl    = 65
    #pa      = 72
    fitsfile = 'testfits/l1527.c18o.contsub.rbp05.mlt100.clean.majpv.fits'
    vsys    = 5.8
    incl    = 85
    pa      = 0.
    #fitsfile = 'testfits/TMC1A_C18O_t2000klam.image.pv_invert.fits'
    #vsys    = 6.4
    #incl    = 48.
    #pa      = 0.

    outname = 'test'
    rms     = 5.e-3 # Jy/beam
    thr     = 6.     # 6 sigma
    dist    = 140.
    clevels = np.array([-3,3,6,9,12,15,20,25,30])*rms
    xlim    = [] # arcsec
    vlim    = []
    Mlim    = []
    # ---------------------


    # -------- main --------
    impv = PVFit(fitsfile, rms, vsys, dist, pa=pa)
    impv.get_edgeridge(outname, thr=thr, xlim=xlim, vlim=vlim,
                       Mlim=Mlim, mode='gauss')
    impv.fit_edgeridge(include_vsys=False, include_dp=False,
                       include_pin=True,
                       filehead='testfit', show_corner=False,
                       minrelerr=0.01, minabserr=0.1)
    print(impv.results)
    impv.plotresults_pvdiagram(clevels=clevels)
    impv.plotresults_rvplane()
    #print(impv.results_sorted)
    # ----------------------



if __name__ == '__main__':
    main()
