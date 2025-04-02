from pvfitting import PVFitting

'-------- INPUTS --------'
pvmajorfits = './pvfitting/IRAS16253_SBLB_C18O_robust_2.0.pvmajor.fits'
pvminorfits = './pvfitting/IRAS16253_SBLB_C18O_robust_2.0.pvminor.fits'
pa_major = 113  # deg
pa_minor = 203  # deg
signmajor = 1  # Positive offset is redshifted --> 1
signminor = 1  # Positive offset is blueshifted --> 1
incl = 65  # deg; 0<= incl <= 90
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
pvfit = PVFitting()
pvfit.put_PV(pvmajorfits=pvmajorfits, pvminorfits=pvminorfits,
             dist=dist, vsys=vsys, rmax=rmax, vmin=vlim[0], vmax=vlim[1],
             sigma=sigma, skipto=5)
pvfit.fit_mockpvd(Mstar_range=[0.01, 1.0],
                  Rc_range=[1, 100],
                  taumax_range=[1e-1, 1e2],
                  frho_range=[1, 1e3],
                  fixed_params={'alphainfall': 1, 'sig_mdl': 0},
                  show=False, filename=filehead, vmask=vmask,
                  incl=incl, pa_major=pa_major, pa_minor=pa_minor,
                  kwargs_emcee_corner={'nwalkers_per_ndim': 4,
                                       'nburnin': 30,
                                       'nsteps': 70,
                                       'rangelevel': 0.99},
                  signmajor=signmajor, signminor=signminor,
                  n_nest=[2] * 6, reslim=10, zmax=1000)
pvfit.modeltofits(**pvfit.popt, filehead=filehead)
'-------------------------------------'