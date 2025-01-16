from pvfitting import PVFitting

'-------- INPUTS --------'
pvmajorfits = './pvfitting/IRAS16253_SBLB_C18O_robust_2.0.pvmajor.fits'
pvminorfits = './pvfitting/IRAS16253_SBLB_C18O_robust_2.0.pvminor.fits'
pa = 113 - 180  # deg
incl = 65  # deg
vsys = 4  # km/s
dist = 139  # pc
sigma = 1.9e-3  # Jy/beam; None means automatic calculation.
rmax = 200  # au
vlim = (-3, 3)
vmask = (-0.2, 0.4)
show_figs = True
'------------------------'

'-------- HOW TO DO EACH STEP --------'
filehead = pvmajorfits.replace('.pvmajor.fits', '')
pvsil = PVFitting()
pvsil.put_PV(pvmajorfits=pvmajorfits, pvminorfits=pvminorfits,
             dist=dist, vsys=vsys, rmax=rmax, vmin=vlim[0], vmax=vlim[1],
             sigma=sigma)
pvsil.fit_mockpvd(Mstar_range=[0.01, 1.0],
                  Rc_range=[1, 100],
                  taumax_range=[1e-2, 1e2],
                  frho_range=[1, 1e3],
                  fixed_params={'alphainfall': 1, 'sig_mdl': 0},
                  show=False, filename=filehead, vmask=vmask,
                  incl=incl, pa_maj=pa, pa_min=pa+90.,
                  kwargs_emcee_corner={'nwalkers_per_ndim':4,
                                       'nburnin':100,
                                       'nsteps':400,
                                       'rangelevel': 0.99},
                  signmajor=None, signminor=None,
                  n_nest=[2] * 5, reslim=10,
                  log=True, zmax=1000)
'-------------------------------------'