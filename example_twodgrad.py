from twodgrad import TwoDGrad

'-------- INPUTS --------'
cubefits = './twodgrad/testcube.fits'
center = '16h28m21.615s -24d36m23.33s'
pa = 0  # deg
incl = 45  # deg
vsys = 0  # km/s
dist = 140  # pc
sigma = 2e-3  # Jy/beam; None means automatic calculation.
cutoff = 5.0  # sigma
xmax = 1.5 * dist  # au
ymax = xmax  # au
vmax = 4.0  # km/s
vmin = -vmax  # km/s
vmask = [-2, 2]  # km/s
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
                  vmin=vmin, vmax=vmax)
tdg.get_2Dcenter(cutoff=cutoff, vmask=vmask,
                 minrelerr=minrelerr, minabserr=minabserr, method=method)
tdg.filtering(pa0=pa, fixcenter=True, axisfilter=False, lowvelfilter=False)
tdg.calc_mstar(incl=incl, voff_fixed=0)
tdg.plot_center(filehead=filehead, pa=pa, show_figs=show_figs)
'-------------------------------------'