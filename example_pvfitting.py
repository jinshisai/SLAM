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
vlim = (-6, 6)
vmask = (-0.5, 0.5)
show_figs = True
'------------------------'

'-------- HOW TO DO EACH STEP --------'
filehead = pvmajorfits.replace('.pvmajor.fits', '')
pvsil = PVFitting()
pvsil.put_PV(pvmajorfits=pvmajorfits, pvminorfits=pvminorfits,
             dist=dist, vsys=vsys, rmax=rmax, vmin=vlim[0], vmax=vlim[1],
             sigma=sigma)
pvsil.check_modelgrid(nsubgrid=1, n_nest=[2] * 3, reslim=10)
pvsil.fit_mockpvd(incl=incl,
                  Mstar_range=[0.03, 0.3],
                  Rc_range=[10, 100],
                  taumax_range=[1e-2, 1e4],
                  frho_range=[1, 1e3],
                  fixed_params={'alphainfall': 1, 'sig_mdl': 0},
                  show=False, filename=filehead, vmask=vmask,
                  pa_maj=pa, pa_min=pa+90.,
                  kwargs_emcee_corner={'nwalkers_per_ndim':2,
                                       'nburnin':100,
                                       'nsteps':200,
                                       'rangelevel': 0.95},
                  signmajor=None, signminor=None,
                  nsubgrid=1, n_nest=[2] * 3, reslim=10,
                  set_title=False, log=True,
                  zmax=1000)
'-------------------------------------'