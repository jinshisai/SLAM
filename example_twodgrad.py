from twodgrad import TwoDGrad

'-------- INPUTS --------'
cubefits = './twodgrad/IRAS16253_SBLB_C18O_robust_2.0.imsub.fits'
center = '16h28m21.61526785s -24d36m24.32538414s'
pa = 113  # deg
incl = 65  # deg
vsys = 4  # km/s
dist = 139  # pc
sigma = 1.9e-3  # Jy/beam; None means automatic calculation.
cutoff = 5.0  # sigma
xmax = 1 * dist  # au
ymax = xmax  # au
vmax = 2.5  # km/s
vmin = -vmax  # km/s
vmask = [-0.5, 0.5]  # km/s
show_figs = True
minrelerr = 0.01
minabserr = 0.1
method = 'mean'  # mean or gauss
'------------------------'


'-------- HOW TO DO EACH STEP --------'
filehead = cubefits.replace('.fits', '')
tdg = TwoDGrad()
tdg.read_cubefits(cubefits=cubefits, center=center,
                  vsys=vsys, dist=dist, sigma=sigma,
                  xmin=-xmax, xmax=xmax,
                  ymin=-ymax, ymax=ymax,
                  vmin=vmin, vmax=vmax,
                  centering_velocity=True)
tdg.get_2Dcenter(cutoff=cutoff, vmask=vmask,
                 minrelerr=minrelerr, minabserr=minabserr, method=method)
tdg.filtering(pa0=pa)
tdg.calc_mstar(incl=incl)
tdg.plot_center(filehead=filehead, pa=pa, show_figs=show_figs)
'-------------------------------------'