from pvfitting import PVFitting

'-------- INPUTS --------'
pvmajorfits = './testfits/test.pvfitting.major.fits'
pvminorfits = './testfits/test.pvfitting.minor.fits'
pa_major = 2.0 - 180  # deg
pa_minor = 92.0 - 180  # deg
signmajor = -1  # Positive offset is redshifted --> 1
signminor = 1  # Positive offset is blueshifted --> 1
incl = 85  # deg; 0<= incl <= 90
vsys = 5.9  # km/s
dist = 140  # pc
sigma = 1.7e-3  # Jy/beam; None means automatic calculation.
rmax = 200  # au
vlim = [-3.6, 3.6]
vmask = [-0.3, 0.3]
show_figs = True
'------------------------'

'-------- HOW TO DO EACH STEP --------'
filehead = 'test.pvfitting'
pvfit = PVFitting()
pvfit.put_PV(pvmajorfits=pvmajorfits, pvminorfits=pvminorfits,
             dist=dist, vsys=vsys, rmax=rmax, vmin=vlim[0], vmax=vlim[1],
             sigma=sigma, skipto=5)
pvfit.fit_mockpvd(Mstar_range=[0.1, 1.0],
                  Rc_range=[30, 300],
                  taumax_range=[1e-2, 1e4],
                  frho_range=[1, 1e3],
                  fixed_params={'alphainfall': 0.6, 'sig_mdl': 5},
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