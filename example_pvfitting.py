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
sigma = 1.7e-3  # Jy/beam
rmax = 200  # au; The fitted area is [-rmax, ramx] x [-rmax, rmax].
vlim = [-3.6, 3.6]  # km/s; relative to vsys
vmask = [-0.3, 0.3]  # km/s; relative to vsys. To exclude channels with large missing flux
'------------------------'

'-------- HOW TO DO EACH STEP --------'
filehead = 'test.pvfitting'
pvfit = PVFitting()
pvfit.put_PV(pvmajorfits=pvmajorfits, pvminorfits=pvminorfits,
             dist=dist, vsys=vsys, rmax=rmax, vmin=vlim[0], vmax=vlim[1],
             sigma=sigma, skipto=5)
pvfit.fit_mockpvd(Mstar_range=[0.1, 1.0],  # Msun; stellar mass
                  Rc_range=[30, 300],  # au; disk radius (=centrifugal radius)
                  taumax_range=[1e-2, 1e4],  # maximum optical depth
                  frho_range=[1, 1e3],  # density jump at (R,z)=(Rc,0). 1 means no jump. A larger value means a higher jump.
                  fixed_params={'alphainfall': 0.6,  # a scaling factor for the radial infall velocity
                                'sig_mdl': 5  # sigma; model uncertainty. Chi squared uses sigma^2 * (1 + sig_mdl^2).
                                },
                  show=False, filename=filehead, vmask=vmask,
                  incl=incl, pa_major=pa_major, pa_minor=pa_minor,
                  kwargs_emcee_corner={'nwalkers_per_ndim': 4,
                                       'nburnin': 30,
                                       'nsteps': 70,
                                       'rangelevel': 0.99},
                  signmajor=signmajor, signminor=signminor,
                  n_nest=[2, 2, 2, 2, 2, 2], # Refinement factors for each nesting level of the nested grid; pixel resolution increases by the given factor at each level.
                  reslim=10, # Resolution threshold that triggers nesting; defines the spatial extent of the next-level subgrid.
                  zmax=1000  # au; length along the line of sight, from -zmax to zmax.
                  )
pvfit.modeltofits(**pvfit.popt, filehead=filehead)
'-------------------------------------'