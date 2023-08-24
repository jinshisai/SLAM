import numpy as np
from pvsilhouette import PVSilhouette

'-------- INPUTS --------'
cubefits = './pvsilhouette/IRAS16253_SBLB_C18O_robust_2.0.imsub.fits'
pvmajorfits = './pvsilhouette/IRAS16253_SBLB_C18O_robust_2.0.pvmajor.fits'
pvminorfits = './pvsilhouette/IRAS16253_SBLB_C18O_robust_2.0.pvminor.fits'
center = '16h28m21.61526785s -24d36m24.32538414s'
pa = 113 - 180  # deg
incl = 65  # deg
vsys = 4  # km/s
dist = 139  # pc
sigma = 1.9e-3  # Jy/beam; None means automatic calculation.
cutoff = 5.0  # sigma
rmax = 200  # au
vlim = (-6, 6)
vmask = (-0.5, 0.5)
show_figs = True
'------------------------'

'-------- HOW TO DO EACH STEP --------'
filehead = pvmajorfits.replace('.pvmajor.fits', '')
pvsil = PVSilhouette()
#pvsil.get_PV(cubefits=cubefits, center=center, pa=pa,
#             vsys=vsys, dist=dist, sigma=sigma,
#             rmax=rmax, vmax=vmax, show=False)
pvsil.put_PV(pvmajorfits=pvmajorfits, pvminorfits=pvminorfits,
             dist=dist, vsys=vsys, rmax=rmax, vmin=vlim[0], vmax=vlim[1],
             sigma=sigma)
pvsil.fitting(incl=incl, Mstar_range=[0.01, 10], Rc_range=[1, 1000],
              alphainfall_range=[0.01, 1],
              cutoff=5, show=show_figs, figname=filehead, vmask=vmask)
'-------------------------------------'