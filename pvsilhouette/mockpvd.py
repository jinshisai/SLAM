import numpy as np
from astropy import constants, units
from scipy.signal import convolve

from pvsilhouette.grid import Nested3DGrid
from pvsilhouette.precalculation import XYZ2rtp
from pvsilhouette import precalculation

au = units.au.to('m')
GG = constants.G.si.value
M_sun = constants.M_sun.si.value
deg = units.deg.to('radian')



class MockPVD(object):
    """
    MockPVD is a class to generate mock PV diagram.

    """
    def __init__(self, x:np.ndarray, z:np.ndarray, v:np.ndarray, 
        nnest:list = None, nsubgrid:int = 1, 
        xlim:list = None, ylim: list = None, zlim:list = None,
        beam:list = None, reslim:float = 5):
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

        # make grid
        self.makegrid(xlim, ylim, zlim, reslim = reslim)
        #print(self.grid.xnest)
        self.xx, self.vv = np.meshgrid(self._x, self.v)



    def generate_mockpvd(self, 
        Mstar: float, Rc: float, alphainfall: float = 1., 
        fflux: float = 1., frho: float = 1., ftau: float = 1.,
        incl: float = 89., withkepler: bool = True, 
        pls: float = -1., plh: float = 0.25, h0: float = 0.1, 
        pa: float | list = 0., linewidth: float = None, 
        rin: float = 1., rout: float = None,
        axis: str = 'both', tauscale: float = 1.,):
        '''
        Generate a mock PV diagram.

        Parameters
        ----------
        Mstar (float): Stellar mass (Msun)
        Rc (float): Centrifugal radius (au)
        alphainfall (float): Decelerating factor
        fflux (float): Factor to scale mock flux
        frho (float): Factor to scale density contrast between disk and envelope
        ftau (float): Factor to scale mock optical depth
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
                _rho, _vlos = self.build(Mstar, Rc, incl, 
                    alphainfall = alphainfall, withkepler = withkepler, 
                    pls = pls, plh = plh, h0 = h0, frho = frho, rin = rin, rout = rout, 
                    axis = _axis, collapse = False, normalize = False)
                rho.append(_rho)
                vlos.append(_vlos)
            # density normalization
            rho_max = np.nanmax(
                [np.nanmax([np.nanmax(i) for i in _rho]) for _rho in rho])
            rho = [ [i / rho_max for i in _rho] for _rho in rho] # normalized rho
            # get PV diagrams
            for _rho, _vlos, _pa in zip(rho, vlos, pa):
                # PV cut
                I_pv = self.generate_pvd(_rho, _vlos, fflux, ftau, beam = self.beam,
                    linewidth = linewidth, pa = _pa)
                I_out.append(I_pv)
            return I_out
        else:
            # build model
            rho, vlos = self.build(Mstar, Rc, incl, 
                alphainfall = alphainfall, withkepler = withkepler, 
                pls = pls, plh = plh, h0 = h0, frho = frho, rin = rin, rout = rout, 
                axis = axis, collapse = False)
            # PV cut
            return self.generate_pvd(rho, vlos, fflux, ftau, beam = self.beam,
                linewidth = linewidth)


    def subgrid(self, axes:list, nsubgrid:int):
        axes_out = []
        for x in axes:
            nx = len(x)
            dx = x[1] - x[0]
            x_e = np.linspace( x[0] - 0.5 * dx, x[-1] + 0.5 * dx, nx*nsubgrid + 1)
            x = 0.5 * (x_e[:-1] + x_e[1:])
            axes_out.append(x)
        return axes_out


    def makegrid(self, 
        xlim:list = None, ylim:list = None, zlim:list = None,
        reslim: float = 5):
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
            grid = Nested3DGrid(x, y, z, xlim, ylim, zlim,
                self.nnest, nlevels = len(self.nnest), reslim = reslim)
        else:
            grid = Nested3DGrid(x, y, z, None, None, None, [1],
                nlevels = 0)
        self.grid = grid


    def gridinfo(self):
        self.grid.gridinfo(units = ['au', 'au', 'au'])


    def build(self, Mstar:float, Rc:float, incl:float,
        alphainfall:float = 1., withkepler: bool = True, 
        pls: float = -1., plh: float = 0.25, h0: float = 0.1, frho:float = 1.,
        rin:float = 1.0, rout:float = None, collapse = False, normalize = True,
        axis: str = 'major'):
        # parameters/units
        irad = np.radians(incl)
        vunit = np.sqrt(GG * Mstar * M_sun / Rc / au) * 1e-3
        elos = [0, -np.sin(irad), np.cos(irad)] # line of sight vector

        # for each nested level
        d_rho = [None] * self.grid.nlevels
        d_vlos = [None] * self.grid.nlevels
        for l in range(self.grid.nlevels):
            X = self.grid.xnest[l] / Rc
            Y = self.grid.ynest[l] / Rc
            Z = self.grid.znest[l] / Rc
            if precalculation.elos_r[axis][l] is None:
                # along which axis
                if axis == 'major':
                    r, t, p = XYZ2rtp(irad, 0, X, Y, Z)
                else:
                    r, t, p = XYZ2rtp(irad, 0, Y, X, Z)
                precalculation.update(t, p, irad, axis, l)
            else:
                r = np.linalg.norm([X, Y, Z], axis=0)
            # get density and velocity
            rho = precalculation.get_rho(r, frho, axis, l)
            #if len(rho.shape) != 3: rho = rho.reshape(nx, ny, nz) # in 3D
            vlos = precalculation.get_vlos(r, alphainfall, axis, l) * vunit
            #if len(vlos.shape) != 3: vlos = vlos.reshape(nx, ny, nz) # in 3D

            # inner and outer edge
            rho[r * Rc <= rin] = 0.
            if rout is not None: rho[np.where(r * Rc > rout)] = np.nan

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
        fflux:float = 1., ftau:float = 1., beam:list = None, linewidth: float = None, 
        pa: float = 0.):
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        # integrate along Z axis
        v = self.v.copy()
        nv = len(v)
        delv = v[1] - v[0]
        ve = np.hstack([v - delv * 0.5, v[-1] + 0.5 * delv])


        if type(rho) == np.ndarray: rho = [rho.ravel()]
        if type(vlos) == np.ndarray: vlos = [vlos.ravel()]

        #tau_v  = np.zeros((nv, ny, nx)) # mock tau
        # go up from the deepest layer to the upper layer
        rho_col = [self.grid.collapse(rho, upto = l) for l in range(self.grid.nlevels)]
        vlos_col = [self.grid.collapse(vlos, upto = l) for l in range(self.grid.nlevels)]

        # binning
        def binning(data, nbin):
            d_avg = np.array([
                data[:, i::nbin, i::nbin]
                for i in range(nbin)
                ])
            return np.nanmean(d_avg, axis = 0)

        # innermost grid
        _nx, _ny, _nz = self.grid.ngrids[-1] # dimension of the l-th layer
        rho_l = rho_col[-1].reshape(_nx, _ny, _nz)
        vlos_l = vlos_col[-1].reshape(_nx, _ny, _nz)
        tau_v = np.zeros((nv, _ny, _nx))
        for i in range(nv):
            _rho = np.where((ve[i] <= vlos_l) & (vlos_l < ve[i+1]), rho_l, np.nan)
            tau_v[i, :, :] = np.nansum(_rho, axis=2).T * ftau

        # if nested grid
        if self.grid.nlevels >= 2:
            for l in range(self.grid.nlevels-2,-1,-1):
                _nx, _ny, _nz = self.grid.ngrids[l] # dimension of the l-th layer
                rho_l = rho_col[l].reshape(_nx, _ny, _nz)
                vlos_l = vlos_col[l].reshape(_nx, _ny, _nz)

                if l < (self.grid.nlevels - 1):
                    # starting & ending indices of the inner grid
                    ximin, ximax = self.grid.xinest[l+1]
                    yimin, yimax = self.grid.yinest[l+1]
                    zimin, zimax = self.grid.zinest[l+1]
                    rho_l[ximin:ximax+1, yimin:yimax+1, zimin:zimax+1] = 0.

                tau_vl = np.zeros((nv, _ny, _nx))
                for i in range(nv):
                    _rho = np.where((ve[i] <= vlos_l) & (vlos_l < ve[i+1]), rho_l, np.nan)
                    tau_vl[i, :, :] = np.nansum(_rho, axis=2).T * ftau

                # add values from the inner grid
                _tau_v = binning(tau_v, self.grid.nsub[l])
                tau_vl[:, yimin:yimax+1, ximin:ximax+1] += _tau_v
                tau_v = tau_vl

        # convolution along the spectral direction
        if linewidth is not None:
            #gaussbeam = np.exp(- 0.5 * (v /(linewidth / 2.35))**2.)
            gaussbeam = np.exp(- (v - v[nv//2 - 1 + nv%2])**2. / linewidth**2.)
            gaussbeam /= np.sum(gaussbeam)
            tau_v = convolve(tau_v, 
            np.array([[gaussbeam]]).T, mode='same') # conserve integrated value

        I_cube = fflux * ( 1. - np.exp(-tau_v) ) # fscale = (Bv(Tex) - Bv(Tbg))

        # beam convolution
        if beam is not None:
            bmaj, bmin, bpa = beam
            xb, yb = rot(*np.meshgrid(self.x, self.y, indexing='ij'), np.radians(bpa - pa))
            gaussbeam = np.exp(
                - 0.5 * (yb /(bmin / 2.35))**2. \
                - 0.5 * (xb /(bmaj / 2.35))**2.)
            gaussbeam /= np.sum(gaussbeam)
            I_cube = convolve(I_cube, np.array([gaussbeam]), mode='same')

        # output
        I_pv = I_cube[:,ny//2,:]

        if self.nsubgrid > 1:
            I_pv = np.nanmean(
                np.array([
                I_pv[:, i::self.nsubgrid]
                for i in range(self.nsubgrid)
                ]),
                axis = 0)
        return I_pv


def rot(x, y, pa):
    s = x * np.cos(pa) - y * np.sin(pa)  # along minor axis
    t = x * np.sin(pa) + y * np.cos(pa)  # along major axis
    return np.array([s, t])
