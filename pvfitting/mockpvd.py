import numpy as np
from astropy import constants, units
from scipy.signal import convolve

from pvfitting.grid import Nested3DGrid
from pvfitting.precalculation import XYZ2rtp
from pvfitting import precalculation
from pvfitting.precalculation import rho2tau

au = units.au.to('m')
GG = constants.G.si.value
M_sun = constants.M_sun.si.value
deg = units.deg.to('radian')



class MockPVD(object):
    """
    MockPVD is a class to generate mock PV diagram.

    """
    def __init__(self, x:np.ndarray, z:np.ndarray, v:np.ndarray, 
                 nnest:list | None = None, nsubgrid: int = 1, 
                 xlim:list | None = None, ylim: list | None = None, zlim:list | None = None,
                 beam:list | None = None, reslim: float = 5,
                 signmajor: int = 1, signminor: int = 1,
                 pa_major: float = 0, pa_minor: float = 90):
        '''
        Initialize MockPVD with a given grid. z is the line of sight axis.

        Parameters
        ----------
        x, z, v (array): 1D arrays for x, z and v axes.
        nsubgrid (int): Number of sub pixels to which the original pixel is divided.
        nnest (list): Number of sub pixels of the nested grid, to which the parental pixel is divided.
         E.g., if nnest=[4], a nested grid having a resolution four times finer than the resolution
         of the parental grid is created. If [4, 4], the grid is nested to two levels and 
         each has a four times better resolution than its parental grid.
        xlim, zlim (list): x and z ranges for the nested grid. Must be given as [[xmin0, xmax0], [xmin1, xmax1]].
        beam (list): Beam info, which must be give [major, minor, pa].
        signmajor: 1 for the case where the positive offset is on the redshifted side in the PV diagram along the major axis; otherwise -1.
        signminor: 1 for the case where the negative offset is on the redshifted side in the PV diagram along the minor axis; otherwise -1.
        pa_major: PA of the positive offset of the PV diagram along the major axis.
        pa_minor: PA of the positive offset of the PV diagram along the minor axis.
        '''
        super(MockPVD, self).__init__()

        # save input
        self._x, self._z = x, z
        self._nx, self._nz = len(x), len(z)
        # subgrid
        self.nsubgrid = nsubgrid
        if nsubgrid > 1:
            x, z = self.subgrid([x, z], nsubgrid)
            self.x, self.z = x, z
        else:
            self.x, self.z = x, z
        self.nx, self.nz = len(x), len(z)
        self.v = v
        # nested grid
        self.nnest = nnest
        # beam
        self.beam = beam
        self.pa_major = pa_major
        self.pa_minor = pa_minor
        pa_major_red = pa_major + (0. if signmajor > 0 else 180.)
        pa_minor_blue = pa_minor + (0. if signminor > 0 else 180.)
        dpa = np.radians(pa_minor_blue - pa_major_red)
        dpa = np.angle(np.exp(1j * dpa))
        self.iradshift = 0 if dpa > 0 else np.pi
        self.signmajor = signmajor if dpa > 0 else -signmajor
        self.signminor = signminor

        # make grid
        self.makegrid(xlim, ylim, zlim, reslim = reslim)
        #print(self.grid.xnest)
        self.xx, self.vv = np.meshgrid(self._x, self.v)



    def generate_mockpvd(self, Mstar:float, Rc:float, alphainfall: float = 1., 
                         taumax: float = 1., frho: float = 1.,
                         incl: float = 89., pa: float | list = 0.,
                         linewidth: float | None = None, rin: float = 1., 
                         rout: float | None = None, axis: str = 'both'):
        '''
        Generate a mock PV diagram.

        Parameters
        ----------
        Mstar (float): Stellar mass (Msun)
        Rc (float): Centrifugal radius (au)
        alphainfall (float): Decelerating factor
        taumax (float): Factor to scale mock optical depth
        frho (float): Factor to scale density contrast between disk and envelope
        incl (float): Inclination angle (deg). Incl=90 deg corresponds to edge-on configuration.
        axis (str): Axis of the pv cut. Must be major, minor or both.
        '''

        # check
        if axis not in ['major', 'minor', 'both']:
            print("ERROR\tgenerate_mockpvd: axis input must be 'major', 'minor' or 'both'.")
            return 0

        # Generate PV diagram
        if axis == 'both':
            I_out = []
            rho = []
            vlos = []
            # build model along major and minor axes
            for _axis in ['major', 'minor']:
                # build model
                _rho, _vlos = self.build(Mstar=Mstar, Rc=Rc, incl=incl,
                                         alphainfall=alphainfall, frho=frho,
                                         rin=rin, rout=rout, axis=_axis,
                                         collapse=False, normalize=False)
                rho.append(_rho)
                vlos.append(_vlos)
            # density normalization
            rho_max = np.nanmax([np.nanmax([np.nanmax(i) for i in _rho]) for _rho in rho])
            rho = [ [i / rho_max for i in _rho] for _rho in rho] # normalized rho
            # get PV diagrams
            palist = [self.pa_major, self.pa_minor]
            signlist = [self.signmajor, self.signminor]
            for _rho, _vlos, _pa, _sign in zip(rho, vlos, palist, signlist):
                # PV cut
                I_pv = self.generate_pvd(rho=_rho, vlos=_vlos, taumax=taumax,
                                         beam=self.beam, linewidth=linewidth, pa=_pa)
                I_pv = I_pv[:, ::_sign]
                I_out.append(I_pv)
            return I_out
        else:
            # build model
            rho, vlos = self.build(Mstar=Mstar, Rc=Rc, incl=incl,
                                   alphainfall=alphainfall, frho=frho,
                                   rin=rin, rout=rout, axis=axis,
                                   collapse=False)
            # PV cut
            return self.generate_pvd(rho=rho, vlos=vlos, taumax=taumax,
                                     beam=self.beam, linewidth=linewidth, pa=pa)


    def subgrid(self, axes:list, nsubgrid:int):
        axes_out = []
        for x in axes:
            nx = len(x)
            dx = x[1] - x[0]
            x_e = np.linspace( x[0] - 0.5 * dx, x[-1] + 0.5 * dx, nx*nsubgrid + 1)
            x = 0.5 * (x_e[:-1] + x_e[1:])
            axes_out.append(x)
        return axes_out


    def makegrid(self, xlim: list | None = None, ylim: list | None = None,
                 zlim: list | None = None, reslim: float = 5):
        # parental grid
        # x and z
        x = self.x
        z = self.z
        dx = x[1] - x[0]
        # y axis
        if self.beam is not None:
            bmaj, bmin, bpa = self.beam
            y = np.arange(
                -int(bmaj / dx * 3. / 2.35) -1, 
                int(bmaj / dx * 3. / 2.35) + 2, 
                1) * dx # +/- 3 sigma
            self.y = y
        else:
            y = np.array([-dx, 0., dx])


        if self.nnest is not None:
            grid = Nested3DGrid(x, y, z, xlim, ylim, zlim, self.nnest,
                                nlevels=len(self.nnest), reslim=reslim)
        else:
            grid = Nested3DGrid(x, y, z, None, None, None, [1], nlevels=0)
        self.grid = grid


    def gridinfo(self):
        self.grid.gridinfo(units=['au', 'au', 'au'])


    def build(self, Mstar:float, Rc:float, incl:float,
              alphainfall: float = 1., frho: float = 1.,
              rin: float = 1.0, rout: float | None = None,
              collapse: bool = False, normalize: bool = True,
              axis: str = 'major'):
        # parameters/units
        irad = np.abs(np.asin(np.sin(np.radians(incl)))) + self.iradshift
        vunit = np.sqrt(GG * Mstar * M_sun / Rc / au) * 1e-3

        # for each nested level
        d_rho = [None] * self.grid.nlevels
        d_vlos = [None] * self.grid.nlevels
        for l in range(self.grid.nlevels):
            if precalculation.elos_r[axis][l] is None:
                X = self.grid.xnest[l] / Rc
                Y = self.grid.ynest[l] / Rc
                Z = self.grid.znest[l] / Rc
                # along which axis
                if axis == 'major':
                    r, t, p = XYZ2rtp(irad, 0, X, Y, Z)
                else:
                    r, t, p = XYZ2rtp(irad, 0, -Y, X, Z)
                precalculation.update(r * Rc, t, p, irad, axis, l)
            r_org = precalculation.r_org[axis][l]
            # get density and velocity
            rho, vlos = precalculation.get_rho_vlos(Rc, frho, alphainfall, axis, l)
            vlos = vlos * vunit
            #if len(rho.shape) != 3: rho = rho.reshape(nx, ny, nz) # in 3D
            #if len(vlos.shape) != 3: vlos = vlos.reshape(nx, ny, nz) # in 3D

            # inner and outer edge
            rho[r_org <= rin] = 0.
            if rout is not None: rho[np.where(r_org > rout)] = 0

            d_rho[l] = rho
            d_vlos[l] = vlos

        # normalize
        if normalize:
            rho_max = np.nanmax([np.nanmax(i) for i in d_rho])
            d_rho = [i / rho_max for i in d_rho]

        # collapse
        if collapse:
            if self.grid.nlevels >= 2:
                d_rho = self.grid.collapse(d_rho)
                d_vlos = self.grid.collapse(d_vlos)
            else:
                d_rho = d_rho[0]
                d_vlos = d_vlos[0]
            d_rho = d_rho.reshape(self.grid.nx, self.grid.ny, self.grid.nz)
            d_vlos = d_vlos.reshape(self.grid.nx, self.grid.ny, self.grid.nz)

        return d_rho, d_vlos



    def generate_pvd(self, rho:np.ndarray | list, vlos:np.ndarray | list, 
                     taumax: float = 1., beam: list | None = None,
                     linewidth: float | None = None, pa: float = 0.):
        ny = self.grid.ny
        # integrate along Z axis
        v = self.v.copy()
        nv = len(v)
        delv = v[1] - v[0]
        if precalculation.vedge is None:
            precalculation.vedge = np.hstack([v - delv * 0.5, v[-1] + 0.5 * delv])

        if type(rho) == np.ndarray: rho = [rho.ravel()]
        if type(vlos) == np.ndarray: vlos = [vlos.ravel()]

        #tau_v  = np.zeros((nv, ny, nx)) # mock tau
        # go up from the deepest layer to the upper layer
        rho_col = [self.grid.collapse(rho, upto=l) for l in range(self.grid.nlevels)]
        vlos_col = [self.grid.collapse(vlos, upto=l) for l in range(self.grid.nlevels)]

        # binning
        def binning(data, nbin):
            d_avg = np.array([
                data[:, i::nbin, i::nbin]
                for j in range(nbin) for i in range(nbin)
                ])
            return np.nanmean(d_avg, axis=0)

        # innermost grid
        _nx, _ny, _nz = self.grid.ngrids[-1] # dimension of the l-th layer
        rho_l = rho_col[-1].reshape(_nx, _ny, _nz)
        vlos_l = vlos_col[-1].reshape(_nx, _ny, _nz)
        dz = self.grid.zaxes[-1][1] - self.grid.zaxes[-1][0]
        tau_v = rho2tau(vlos_l, rho_l) * dz
        # if nested grid
        if self.grid.nlevels >= 2:
            for l in range(self.grid.nlevels-2,-1,-1):
                _nx, _ny, _nz = self.grid.ngrids[l] # dimension of the l-th layer
                rho_l = rho_col[l].reshape(_nx, _ny, _nz)
                vlos_l = vlos_col[l].reshape(_nx, _ny, _nz)
                dz = self.grid.zaxes[l][1] - self.grid.zaxes[l][0]

                if l < (self.grid.nlevels - 1):
                    # starting & ending indices of the inner grid
                    ximin, ximax = self.grid.xinest[l+1]
                    yimin, yimax = self.grid.yinest[l+1]
                    zimin, zimax = self.grid.zinest[l+1]
                    rho_l[ximin:ximax+1, yimin:yimax+1, zimin:zimax+1] = 0.

                tau_vl = rho2tau(vlos_l, rho_l) * dz
                # add values from the inner grid
                _tau_v = binning(tau_v, self.grid.nsub[l])
                tau_vl[:, yimin:yimax+1, ximin:ximax+1] += _tau_v
                tau_v = tau_vl

        # convolution along the spectral direction
        if linewidth is not None:
            if precalculation.gauss_v is None:
                #gaussbeam = np.exp(- 0.5 * (v /(linewidth / 2.35))**2.)
                gaussbeam = np.exp(-(v - v[nv//2 - 1 + nv%2])**2. / linewidth**2.)
                gaussbeam /= np.sum(gaussbeam)
                precalculation.gauss_v = gaussbeam[:, np.newaxis, np.newaxis]
            g = precalculation.gauss_v
            tau_v = convolve(tau_v, g, mode='same') # conserve integrated value

        I_cube = 1. - np.exp(-tau_v / np.nanmax(tau_v) * taumax)

        # beam convolution
        if beam is not None:
            if precalculation.gauss_xy is None:
                bmaj, bmin, bpa = beam
                xb, yb = rot(*np.meshgrid(self.x, self.y), -np.radians(bpa - pa))
                gaussbeam = np.exp(-4*np.log(2) * ((yb/bmin)**2 + (xb/bmaj)**2))
                gaussbeam /= np.sum(gaussbeam)
                precalculation.gauss_xy = gaussbeam[np.newaxis, :, :]
            g = precalculation.gauss_xy
            I_cube = convolve(I_cube, g, mode='same')

        # output
        I_pv = I_cube[:, ny//2, :]

        if self.nsubgrid > 1:
            I_pv = np.nanmean(np.array([I_pv[:, i::self.nsubgrid] 
                                        for i in range(self.nsubgrid)]), axis=0)
        return I_pv


def rot(x, y, pa):
    s = x * np.cos(pa) - y * np.sin(pa)  # along minor axis
    t = x * np.sin(pa) + y * np.cos(pa)  # along major axis
    return np.array([s, t])
