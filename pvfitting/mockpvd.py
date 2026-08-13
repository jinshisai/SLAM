import numpy as np
from astropy import constants, units
from scipy.signal import convolve

from pvfitting.grid import Nested3DGrid
from pvfitting.precalculation import XYZ2rtp
from pvfitting import precalculation
from pvfitting.precalculation import resolve_num_threads, rho2tau
from utils import rot

au = units.au.to('m')
GG = constants.G.si.value
M_sun = constants.M_sun.si.value
deg = units.deg.to('radian')


class MockPVD(object):
    """Generate mock PV diagrams of a protostellar disk-envelope system.

    The model assumes a cut-off disk and the UCM envelope model. A mock
    position-velocity (PV) diagram is calculated from the scaled optical
    depth, obtained by summing the gas density along the line of sight and
    rescaling the integrated value. The normalized intensity follows the
    radiative-transfer equation ``I_v = 1 - exp(-tau_v)``.

    Args:
        x (np.ndarray): One-dimensional spatial coordinates of the PV cut in
            au.
        z (np.ndarray): One-dimensional line-of-sight coordinates in au.
        v (np.ndarray): One-dimensional velocity coordinates in km/s.
        nnest (list or None, optional): Refinement factor for each nested-grid
            level. For example, ``[4, 2]`` creates three total levels—the
            original grid and two nested levels—with successive refinement
            factors of four and two. None disables nested refinement. Defaults
            to None.
        nsubgrid (int, optional): Refinement factor applied to the entire
            original spatial grid before nesting. Defaults to 1.
        xlim (list or None, optional): Per-level x-coordinate ranges for the
            nested grid, formatted as ``[[xmin0, xmax0], ...]``. None derives
            them from ``reslim``. Defaults to None.
        ylim (list or None, optional): Per-level y-coordinate ranges for the
            nested grid, formatted as ``[[ymin0, ymax0], ...]``. None derives
            them from ``reslim``. Defaults to None.
        zlim (list or None, optional): Per-level z-coordinate ranges for the
            nested grid, formatted as ``[[zmin0, zmax0], ...]``. None derives
            them from ``reslim``. Defaults to None.
        beam (list, np.ndarray, or None, optional): Beam major axis, minor
            axis, and position angle ``[major, minor, pa]`` in au, au, and
            degrees, respectively. None disables beam convolution. Defaults
            to None.
        reslim (float, optional): Resolution threshold used instead of exact
            nesting ranges. If nesting is requested and any range is None,
            each such range extends to plus and minus ``reslim`` times the
            parent-level resolution. Defaults to 10.
        signmajor (int, optional): Sign of the rotational line-of-sight
            velocity. +1 makes the positive major-axis side redshifted and -1
            makes it blueshifted. Defaults to 1.
        signminor (int, optional): Sign of the radial-infall line-of-sight
            velocity. +1 makes the positive minor-axis side blueshifted and -1
            makes it redshifted. Defaults to 1.
        pa_major (float, optional): Position angle of the positive major-axis
            offset in degrees. Defaults to 0.
        pa_minor (float, optional): Position angle of the positive minor-axis
            offset in degrees. Defaults to 90.
        num_threads (int, str, or None, optional): Number of Numba threads for
            line-of-sight integration. None uses a conservative automatic
            budget; ``'all'`` uses every thread available to Numba. Defaults
            to None.

    Notes:
        The intensity calculation assumes an isothermal disk and envelope and
        a constant molecular abundance. To compare a normalized mock PV
        diagram with observational data, it is rescaled by the observed
        flux.
    """

    def __init__(self, x: np.ndarray, z: np.ndarray, v: np.ndarray,
                 nnest: list[int] | None = None, nsubgrid: int = 1,
                 xlim: list[list[float]] | None = None,
                 ylim: list[list[float]] | None = None,
                 zlim: list[list[float]] | None = None,
                 beam: list[float] | np.ndarray | None = None,
                 reslim: float = 10,
                 signmajor: int = 1, signminor: int = 1,
                 pa_major: float = 0, pa_minor: float = 90,
                 num_threads: int | str | None = None) -> None:
        """Initialize the model coordinates, orientation, beam, and grid."""
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
        self.num_threads = resolve_num_threads(num_threads)
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
        self.makegrid(xlim, ylim, zlim, reslim=reslim)
        self.xx, self.vv = np.meshgrid(self._x, self.v)

    def generate_mockpvd(self, Mstar: float, Rc: float, alphainfall: float = 1.,
                         taumax: float = 1., frho: float = 1.,
                         incl: float = 89., pa: float = 0.,
                         linewidth: float | None = None, rin: float = 1.,
                         rout: float | None = None,
                         axis: str = 'both'
                         ) -> np.ndarray | list[np.ndarray] | int:
        """Generate a mock PV diagram.

        Args:
            Mstar (float): Stellar mass in solar masses.
            Rc (float): Centrifugal radius in au.
            alphainfall (float, optional): Decelerating factor that scales the
                radial infall velocity. A value of one means no suppression.
                Defaults to 1.
            taumax (float, optional): Factor used to scale the maximum mock
                optical depth. Defaults to 1.
            frho (float, optional): Factor used to scale the density contrast
                between the disk and envelope. Higher values give a higher
                density on the disk side. Defaults to 1.
            incl (float, optional): Inclination angle in degrees. An
                inclination of 90 degrees corresponds to an edge-on
                configuration. Defaults to 89.
            pa (float, optional): Position angle of a single requested PV cut
                in degrees. When ``axis='both'``, the major- and minor-axis
                position angles supplied at initialization are used instead.
                Defaults to 0.
            linewidth (float or None, optional): Intrinsic line width in km/s
                used for convolution along the velocity axis. None disables
                spectral convolution. Defaults to None.
            rin (float, optional): Inner cut-off radius in au. Density at and
                inside this radius is set to zero. Defaults to 1.
            rout (float or None, optional): Outer cut-off radius in au. None
                applies no outer cut-off. Defaults to None.
            axis (str, optional): Axis of the PV cut: ``'major'``, ``'minor'``,
                or ``'both'``. Defaults to ``'both'``.

        Returns:
            np.ndarray, list, or int: Mock PV diagram for a single requested
            axis, or ``[major, minor]`` diagrams when ``axis='both'``. Integer
            ``0`` is returned after an error message if ``axis`` is invalid.
        """

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
            if rho_max != 0:
                rho = [[i / rho_max for i in _rho] for _rho in rho]
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
                                   collapse=False, normalize=True)
            # PV cut
            return self.generate_pvd(rho=rho, vlos=vlos, taumax=taumax,
                                     beam=self.beam, linewidth=linewidth, pa=pa)

    def subgrid(self, axes: list[np.ndarray],
                nsubgrid: int) -> list[np.ndarray]:
        """Uniformly refine one-dimensional cell-center coordinate axes.

        Args:
            axes (list): Coordinate arrays to refine. Each array must be
                uniformly spaced and contain at least two cell centers.
            nsubgrid (int): Number of refined cells placed across each
                original cell.

        Returns:
            list: Refined cell-center arrays in the same order as ``axes``.
                Each output axis contains ``nsubgrid`` times as many cells as
                its input axis while preserving the original outer edges.
        """
        axes_out = []
        for x in axes:
            nx = len(x)
            dx = x[1] - x[0]
            x_e = np.linspace(x[0] - 0.5 * dx, x[-1] + 0.5 * dx, nx*nsubgrid + 1)
            x = 0.5 * (x_e[:-1] + x_e[1:])
            axes_out.append(x)
        return axes_out

    def makegrid(self, xlim: list[list[float]] | None = None,
                 ylim: list[list[float]] | None = None,
                 zlim: list[list[float]] | None = None,
                 reslim: float = 10) -> None:
        """Construct the three-dimensional model grid.

        Args:
            xlim (list or None, optional): Per-level x-coordinate ranges for
                nested refinement. None derives them from ``reslim``. Defaults
                to None.
            ylim (list or None, optional): Per-level y-coordinate ranges for
                nested refinement. None derives them from ``reslim``. Defaults
                to None.
            zlim (list or None, optional): Per-level z-coordinate ranges for
                nested refinement. None derives them from ``reslim``. Defaults
                to None.
            reslim (float, optional): Half-width of an automatically selected
                refinement region, in cells of its parent level. Defaults to
                10.

        Notes:
            The x and z axes are those supplied at initialization, optionally
            refined by ``nsubgrid``. When a beam is available, the y axis
            covers approximately three Gaussian standard deviations on either
            side of zero; otherwise it contains three cells centered on zero.
            The resulting :class:`Nested3DGrid` is stored as ``self.grid``.
        """
        # parental grid
        # x and z
        x = self.x
        z = self.z
        dx = x[1] - x[0]
        # y axis
        if self.beam is not None:
            bmaj, bmin, bpa = self.beam
            y = np.arange(
                - int(bmaj / dx * 3. / 2.35) - 1,
                int(bmaj / dx * 3. / 2.35) + 2,
                1) * dx  # +/- 3 sigma
            self.y = y
        else:
            y = np.array([-dx, 0., dx])

        if self.nnest is not None:
            grid = Nested3DGrid(x, y, z, xlim, ylim, zlim, self.nnest,
                                nlevels=len(self.nnest), reslim=reslim)
        else:
            grid = Nested3DGrid(x, y, z, None, None, None, [1], nlevels=0)
        self.grid = grid

    def gridinfo(self) -> None:
        """Print the model-grid resolutions and limits in au."""
        self.grid.gridinfo(units=['au', 'au', 'au'])

    def build(self, Mstar: float, Rc: float, incl: float,
              alphainfall: float = 1., frho: float = 1.,
              rin: float = 1.0, rout: float | None = None,
              collapse: bool = False, normalize: bool = True,
              axis: str = 'major'
              ) -> (tuple[list[np.ndarray], list[np.ndarray]]
                    | tuple[np.ndarray, np.ndarray]):
        """Build density and line-of-sight velocity fields on the model grid.

        Args:
            Mstar (float): Stellar mass in solar masses.
            Rc (float): Centrifugal radius in au.
            incl (float): Inclination angle in degrees. An inclination of 90
                degrees corresponds to an edge-on configuration.
            alphainfall (float, optional): Scaling applied to the radial
                infall velocity. A value of one means no suppression. Defaults
                to 1.
            frho (float, optional): Density jump at the centrifugal radius.
                Higher values give a higher density on the disk side. Defaults
                to 1.
            rin (float, optional): Inner cut-off radius in au. Density at and
                inside this radius is set to zero. Defaults to 1.
            rout (float or None, optional): Outer cut-off radius in au. None
                applies no outer cut-off. Defaults to None.
            collapse (bool, optional): Whether to average all nested levels
                back onto the original three-dimensional grid. Defaults to
                False.
            normalize (bool, optional): Whether to divide density at every
                level by the maximum density across all levels. Defaults to
                True.
            axis (str, optional): PV-cut axis, ``'major'`` or ``'minor'``.
                Defaults to ``'major'``.

        Returns:
            tuple: Density fields and line-of-sight velocity fields, with the
                latter in km/s. When ``collapse`` is False, each item is a
                list of flattened arrays ordered from the original to the
                innermost grid level. When True, each item is an array with
                shape ``(nx, ny, nz)`` on the original grid.

        Notes:
            Geometry terms that depend only on the grid, inclination, and cut
            axis are cached in :mod:`pvfitting.precalculation` for reuse.
        """
        # parameters/units
        irad = np.abs(np.arcsin(np.sin(np.radians(incl)))) + self.iradshift
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

            # inner and outer edge
            rho[r_org <= rin] = 0.
            if rout is not None:
                rho[np.where(r_org > rout)] = 0

            d_rho[l] = rho
            d_vlos[l] = vlos

        # normalize
        if normalize:
            rho_max = np.nanmax([np.nanmax(i) for i in d_rho])
            if rho_max != 0:
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

    def generate_pvd(self, rho: np.ndarray | list[np.ndarray],
                     vlos: np.ndarray | list[np.ndarray],
                     taumax: float = 1.,
                     beam: list[float] | np.ndarray | None = None,
                     linewidth: float | None = None,
                     pa: float = 0.) -> np.ndarray:
        """Convert density and velocity fields into a mock PV diagram.

        Args:
            rho (np.ndarray or list): Density field, either on one grid or as
                flattened arrays for all nested levels.
            vlos (np.ndarray or list): Line-of-sight velocity field in km/s,
                with the same organization and shapes as ``rho``.
            taumax (float, optional): Maximum scaled optical depth. Defaults to
                1.
            beam (list, np.ndarray, or None, optional): Beam major axis, minor
                axis, and position angle ``[major, minor, pa]`` in au, au, and
                degrees. None disables spatial convolution. Defaults to None.
            linewidth (float or None, optional): Intrinsic line width in km/s
                used for convolution along the velocity axis. None disables
                spectral convolution. Defaults to None.
            pa (float, optional): Position angle of the PV cut in degrees,
                used to rotate the beam onto the model grid. Defaults to 0.

        Returns:
            np.ndarray: Normalized intensity of the mock PV diagram, with
                shape ``(nv, nx)`` on the original velocity and position axes.

        Notes:
            Optical depth is integrated along z from the innermost nested
            level outward. The normalized intensity is calculated as
            ``I_v = 1 - exp(-tau_v)`` after scaling the peak optical depth to
            ``taumax``. Spectral and beam kernels are cached in
            :mod:`pvfitting.precalculation`.
        """
        ny = self.grid.ny
        # integrate along Z axis
        v = self.v.copy()
        nv = len(v)
        delv = v[1] - v[0]
        if precalculation.vedge is None:
            precalculation.vedge = np.hstack([v - delv * 0.5, v[-1] + 0.5 * delv])

        if type(rho) is np.ndarray:
            rho = [rho.ravel()]
        if type(vlos) is np.ndarray:
            vlos = [vlos.ravel()]

        # go up from the deepest layer to the upper layer
        rho_col = [self.grid.collapse(rho, upto=l) for l in range(self.grid.nlevels)]
        vlos_col = [self.grid.collapse(vlos, upto=l) for l in range(self.grid.nlevels)]

        # innermost grid
        _nx, _ny, _nz = self.grid.ngrids[-1]  # dimension of the l-th layer
        rho_l = rho_col[-1].reshape(_nx, _ny, _nz)
        vlos_l = vlos_col[-1].reshape(_nx, _ny, _nz)
        dz = self.grid.zaxes[-1][1] - self.grid.zaxes[-1][0]
        tau_v = rho2tau(vlos_l, rho_l, num_threads=self.num_threads) * dz
        # if nested grid
        if self.grid.nlevels >= 2:
            for l in range(self.grid.nlevels-2, -1, -1):
                _nx, _ny, _nz = self.grid.ngrids[l]  # dimension of the l-th layer
                rho_l = rho_col[l].reshape(_nx, _ny, _nz)
                vlos_l = vlos_col[l].reshape(_nx, _ny, _nz)
                dz = self.grid.zaxes[l][1] - self.grid.zaxes[l][0]

                if l < (self.grid.nlevels - 1):
                    # starting & ending indices of the inner grid
                    ximin, ximax = self.grid.xinest[l+1]
                    yimin, yimax = self.grid.yinest[l+1]
                    zimin, zimax = self.grid.zinest[l+1]
                    rho_l[ximin:ximax+1, yimin:yimax+1, zimin:zimax+1] = 0.

                tau_vl = rho2tau(vlos_l, rho_l,
                                 num_threads=self.num_threads) * dz
                # add values from the inner grid
                _tau_v = binning(tau_v, self.grid.nsub[l])
                tau_vl[:, yimin:yimax+1, ximin:ximax+1] += _tau_v
                tau_v = tau_vl

        # convolution along the spectral direction
        if linewidth is not None:
            if precalculation.gauss_v is None:
                gaussbeam = np.exp(- (v - v[nv//2 - 1 + nv % 2])**2. / linewidth**2.)
                gaussbeam /= np.sum(gaussbeam)
                precalculation.gauss_v = gaussbeam[:, np.newaxis, np.newaxis]
            g = precalculation.gauss_v
            tau_v = convolve(tau_v, g, mode='same')  # conserve integrated value

        tau_max = np.nanmax(tau_v)
        if tau_max == 0:
            I_cube = np.zeros_like(tau_v)
        else:
            I_cube = 1. - np.exp(-tau_v / tau_max * taumax)

        # beam convolution
        if beam is not None:
            if precalculation.gauss_xy is None:
                bmaj, bmin, bpa = beam
                xb, yb = rot(*np.meshgrid(self.x, self.y), -np.radians(bpa - pa))
                gaussbeam = np.exp2(-4 * ((xb / bmaj)**2 + (yb / bmin)**2))
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


# binning
def binning(data: np.ndarray, nbin: int) -> np.ndarray:
    """Downsample the two spatial axes of an optical-depth cube.

    Args:
        data (np.ndarray): Optical-depth cube with shape ``(nv, ny, nx)``.
        nbin (int): Spatial refinement factor.

    Returns:
        np.ndarray: NaN-aware mean of the strided spatial samples, with shape
            ``(nv, ny / nbin, nx / nbin)`` when both spatial dimensions are
            divisible by ``nbin``.

    Notes:
        The current implementation pairs identical stride offsets on the y
        and x axes; it does not combine all ``nbin**2`` offset pairs.
    """
    d_avg = np.array([data[:, i::nbin, i::nbin]
                      for j in range(nbin)
                      for i in range(nbin)])
    return np.nanmean(d_avg, axis=0)
