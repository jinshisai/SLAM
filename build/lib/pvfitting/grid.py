# modules
import numpy as np


class Nested3DGrid(object):
    """Construct and manipulate a Cartesian grid with nested refinements.

    Args:
        x (np.ndarray): Cell-center coordinates along the x axis.
        y (np.ndarray): Cell-center coordinates along the y axis.
        z (np.ndarray): Cell-center coordinates along the z axis.
        xlim (list or None): Per-level x-coordinate limits of the regions to
            refine. None derives the limits from ``reslim``.
        ylim (list or None): Per-level y-coordinate limits of the regions to
            refine. None derives the limits from ``reslim``.
        zlim (list or None): Per-level z-coordinate limits of the regions to
            refine. None derives the limits from ``reslim``.
        nsub (list): Refinement factor for each nested level.
        nlevels (int, optional): Number of nested levels in addition to the
            original grid. Defaults to 1.
        reslim (float, optional): Half-width of an automatically selected
            refinement region, in cells of its parent level. Defaults to 5.

    Notes:
        Coordinate arrays must be one-dimensional, uniformly spaced, and
        contain at least two values. Level zero is the original grid, so the
        instance stores ``nlevels + 1`` total levels. Coordinates covered by a
        child level are removed from the flattened parent-level arrays.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray,
                 xlim: list[list[float]] | None,
                 ylim: list[list[float]] | None,
                 zlim: list[list[float]] | None,
                 nsub: list[int], nlevels: int = 1,
                 reslim: float = 5) -> None:
        """Initialize the original grid and its requested nested levels."""
        super(Nested3DGrid, self).__init__()
        # save axes of the mother grid
        self.x = x
        self.y = y
        self.z = z
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        dz = z[1] - z[0]
        self.dx = dx
        self.dy = dy
        self.dz = dz
        xe = np.hstack([x - self.dx * 0.5, x[-1] + self.dx * 0.5])
        ye = np.hstack([y - self.dy * 0.5, y[-1] + self.dy * 0.5])
        ze = np.hstack([z - self.dz * 0.5, z[-1] + self.dz * 0.5])
        self.xe, self.ye, self.ze = xe, ye, ze
        nz, ny, nx = len(z), len(y), len(x)
        self.nz, self.ny, self.nx = nz, ny, nx
        self.xx, self.yy, self.zz = np.meshgrid(x, y, z, indexing='ij')
        self.Lx, self.Ly, self.Lz = xe[-1] - xe[0], ye[-1] - ye[0], ze[-1] - ze[0]

        # nested grid
        self.nsub = nsub
        self.nlevels = nlevels + 1
        # original 1D axes
        self.xaxes = [None] * (nlevels + 1)
        self.yaxes = [None] * (nlevels + 1)
        self.zaxes = [None] * (nlevels + 1)
        self.xaxes[0], self.yaxes[0], self.zaxes[0] = x, y, z
        # grid sizes
        self.ngrids = [(None, None, None)] * (nlevels + 1)
        self.ngrids[0] = (nx, ny, nz)
        # nested grid
        self.xnest = [None] * (nlevels + 1)
        self.ynest = [None] * (nlevels + 1)
        self.znest = [None] * (nlevels + 1)
        self.xnest[0] = self.xx.ravel()
        self.ynest[0] = self.yy.ravel()
        self.znest[0] = self.zz.ravel()
        # starting and ending indices
        self.xinest = [[None, None]] * (nlevels + 1)
        self.yinest = [[None, None]] * (nlevels + 1)
        self.zinest = [[None, None]] * (nlevels + 1)
        # nest
        if self.nlevels > 1:
            if any([xlim is None, ylim is None, zlim is None]):
                _xlim, _ylim, _zlim = self.get_nestinglim(reslim=reslim)
                if xlim is None:
                    xlim = _xlim
                if ylim is None:
                    ylim = _ylim
                if zlim is None:
                    zlim = _zlim
            for l in range(nlevels):
                self.nest(l+1, xlim[l], ylim[l], zlim[l], nsub[l])
            self.xlim, self.ylim, self.zlim = xlim.copy(), ylim.copy(), zlim.copy()
            self.xlim.insert(0, [xe[0], xe[-1]])
            self.ylim.insert(0, [ye[0], ye[-1]])
            self.zlim.insert(0, [ze[0], ze[-1]])
        else:
            self.xlim = [[xe[0], xe[-1]]]
            self.ylim = [[ye[0], ye[-1]]]
            self.zlim = [[ze[0], ze[-1]]]

    def get_nestinglim(self, reslim: float = 5
                       ) -> tuple[list[list[float]],
                                  list[list[float]],
                                  list[list[float]]]:
        """Calculate symmetric refinement limits for every nested level.

        Args:
            reslim (float, optional): Half-width of each refinement region in
                cells of its parent level. Defaults to 5.

        Returns:
            tuple: Lists of ``[minimum, maximum]`` limits along x, y, and z.
                Each list contains one entry per nested level.
        """
        xlim = []
        ylim = []
        zlim = []
        _dx, _dy, _dz = self.dx, self.dy, self.dz
        for l in range(self.nlevels - 1):
            xlim.append([-_dx * reslim, _dx * reslim])
            ylim.append([-_dy * reslim, _dy * reslim])
            zlim.append([-_dz * reslim, _dz * reslim])
            _dx, _dy, _dz = np.array([_dx, _dy, _dz]) / self.nsub[l]

        return xlim, ylim, zlim

    def get_grid(self, l: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return Cartesian coordinate meshes for one grid level.

        Args:
            l (int): Grid level, where zero denotes the original grid.

        Returns:
            tuple: Three arrays containing the x, y, and z coordinates. Their
                common shape is ``(nx, ny, nz)`` for the selected level.

        Notes:
            If the flattened level has had its child region removed, the full
            coordinate mesh is reconstructed from the saved one-dimensional
            axes.
        """
        _nx, _ny, _nz = self.ngrids[l]
        # if it is not collapsed
        if self.xnest[l].size == _nx * _ny * _nz:
            xx = self.xnest[l].reshape(_nx, _ny, _nz)
            yy = self.ynest[l].reshape(_nx, _ny, _nz)
            zz = self.znest[l].reshape(_nx, _ny, _nz)
        else:
            # else
            x, y, z = self.xaxes[l], self.yaxes[l], self.zaxes[l]
            xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        return xx, yy, zz

    def nest(self, l: int, xlim: list[float], ylim: list[float],
             zlim: list[float], nsub: int) -> None:
        """Create a child grid inside a region of its parent level.

        Args:
            l (int): Child-level index. Its parent is level ``l - 1``.
            xlim (list): Minimum and maximum x coordinates to refine.
            ylim (list): Minimum and maximum y coordinates to refine.
            zlim (list): Minimum and maximum z coordinates to refine.
            nsub (int): Number of child cells placed across each parent cell
                along every axis.

        Notes:
            Parent cells covered by the child region are removed from the
            flattened parent arrays. The full parent axes remain available for
            reconstruction by :meth:`get_grid` and :meth:`collapse`.
        """
        x, y, z = self.xaxes[l-1], self.yaxes[l-1], self.zaxes[l-1]
        ximin, ximax, yimin, yimax, zimin, zimax, x_sub, y_sub, z_sub = \
            nestgrid_3D(x, y, z, xlim, ylim, zlim, nsub)
        self.xinest[l] = [ximin, ximax]  # starting and ending indices on the upper-layer grid
        self.yinest[l] = [yimin, yimax]
        self.zinest[l] = [zimin, zimax]
        self.xaxes[l], self.yaxes[l], self.zaxes[l] = x_sub, y_sub, z_sub

        # upper grid
        _nx, _ny, _nz = self.ngrids[l-1]
        if self.xnest[l-1].size == _nx * _ny * _nz:
            xx = self.xnest[l-1].reshape(_nx, _ny, _nz)
            yy = self.ynest[l-1].reshape(_nx, _ny, _nz)
            zz = self.znest[l-1].reshape(_nx, _ny, _nz)
        else:
            xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
        # devide the upper grid into six sub-regions
        # Region 1:  x from 0 to ximin, all y and z
        R1x = xx[:ximin, :, :].ravel()
        R1y = yy[:ximin, :, :].ravel()
        R1z = zz[:ximin, :, :].ravel()
        # Region 2: x from ximax+1 to nx, all y and z
        R2x = xx[ximax+1:, :, :].ravel()
        R2y = yy[ximax+1:, :, :].ravel()
        R2z = zz[ximax+1:, :, :].ravel()
        # Region 3: x from xi0 to ximax, y from 0 to yimin, and all z
        R3x = xx[ximin:ximax+1, :yimin, :].ravel()
        R3y = yy[ximin:ximax+1, :yimin, :].ravel()
        R3z = zz[ximin:ximax+1, :yimin, :].ravel()
        # Region 4: x from xi0 to ximax, y from yimax+1 to ny, and all z
        R4x = xx[ximin:ximax+1, yimax+1:, :].ravel()
        R4y = yy[ximin:ximax+1, yimax+1:, :].ravel()
        R4z = zz[ximin:ximax+1, yimax+1:, :].ravel()
        # Region 5: x from xi0 to ximax, y from yimin to yimax and z from 0 to zimin
        R5x = xx[ximin:ximax+1, yimin:yimax+1, :zimin].ravel()
        R5y = yy[ximin:ximax+1, yimin:yimax+1, :zimin].ravel()
        R5z = zz[ximin:ximax+1, yimin:yimax+1, :zimin].ravel()
        # Region 6: x from xi0 to ximax, y from yimin to yimax and z from zimax+1 to nz
        R6x = xx[ximin:ximax+1, yimin:yimax+1, zimax+1:].ravel()
        R6y = yy[ximin:ximax+1, yimin:yimax+1, zimax+1:].ravel()
        R6z = zz[ximin:ximax+1, yimin:yimax+1, zimax+1:].ravel()
        self.xnest[l-1] = np.concatenate([R1x, R2x, R3x, R4x, R5x, R6x])  # update
        self.ynest[l-1] = np.concatenate([R1y, R2y, R3y, R4y, R5y, R6y])  # update
        self.znest[l-1] = np.concatenate([R1z, R2z, R3z, R4z, R5z, R6z])  # update

        # child grid
        xx_sub, yy_sub, zz_sub = np.meshgrid(x_sub, y_sub, z_sub, indexing='ij')
        self.xnest[l] = xx_sub.ravel()
        self.ynest[l] = yy_sub.ravel()
        self.znest[l] = zz_sub.ravel()
        self.ngrids[l] = (len(x_sub), len(y_sub), len(z_sub))

    def collapse(self, d: list[np.ndarray],
                 upto: int | None = None) -> np.ndarray:
        """Average nested data back onto a selected parent grid.

        Args:
            d (list): Flattened data arrays for all grid levels, ordered from
                the original grid to the innermost grid.
            upto (int or None, optional): Level onto which the data are
                collapsed. None collapses through level zero. Defaults to
                None.

        Returns:
            np.ndarray: Data on the selected level with shape
                ``(nx, ny, nz)``.

        Notes:
            Values outside each refined region are restored from the
            corresponding parent-level array. Values inside it are the
            NaN-aware mean of the child cells.
        """
        d_col = d[-1]  # starting from the inner most grid
        lmax = 0 if upto is None else upto
        for l in range(self.nlevels-1, lmax, -1):
            nsub = self.nsub[l-1]
            ximin, ximax = self.xinest[l]
            yimin, yimax = self.yinest[l]
            zimin, zimax = self.zinest[l]
            # collapse data on the inner grid
            _d = self.binning_onsubgrid_layered(d_col.reshape(self.ngrids[l]), nsub)

            # go next layer
            nx, ny, nz = self.ngrids[l-1]  # size of the upper layer
            d_col = np.empty((nx, ny, nz))
            d_col = np.full((nx, ny, nz), np.nan)

            # insert collapsed data
            d_col[ximin:ximax+1, yimin:yimax+1, zimin:zimax+1] = _d

            # fill upper layer data
            # Region 1: x from zero to ximin, all y and z
            d_col[:ximin, :, :] = \
                d[l-1][:ximin * ny * nz].reshape((ximin, ny, nz))
            # Region 2: x from ximax to nx, all y and z
            i0 = ximin * ny * nz
            i1 = i0 + (nx - ximax - 1) * ny * nz
            d_col[ximax+1:, :, :] = \
                d[l-1][i0:i1].reshape(
                    (nx - ximax - 1, ny, nz))
            # Region 3
            i0 = i1
            i1 = i0 + (ximax + 1 - ximin) * yimin * nz
            d_col[ximin:ximax+1, :yimin, :] = \
                d[l-1][i0:i1].reshape(
                    (ximax + 1 - ximin, yimin, nz))
            # Region 4
            i0 = i1
            i1 = i0 + (ximax + 1 - ximin) * (ny - yimax - 1) * nz
            d_col[ximin:ximax+1, yimax+1:, :] = \
                d[l-1][i0:i1].reshape(
                    (ximax + 1 - ximin, ny - yimax - 1, nz))
            # Region 5
            i0 = i1
            i1 = i0 + (ximax + 1 - ximin) * (yimax + 1 - yimin) * zimin
            d_col[ximin:ximax+1, yimin:yimax+1, :zimin] = \
                d[l-1][i0:i1].reshape(
                    (ximax + 1 - ximin, yimax + 1 - yimin, zimin))
            # Region 6
            i0 = i1
            i1 = i0 + (ximax + 1 - ximin) * (yimax + 1 - yimin) * (nz - zimax - 1)
            d_col[ximin:ximax+1, yimin:yimax+1, zimax+1:] = \
                d[l-1][i0:].reshape((ximax + 1 - ximin,
                                     yimax + 1 - yimin,
                                     nz - zimax - 1))

        return d_col

    def binning_onsubgrid(self, data: np.ndarray) -> np.ndarray:
        """Average a three-dimensional subgrid over its refinement cells.

        Args:
            data (np.ndarray): Three-dimensional data sampled on a refined
                grid.

        Returns:
            np.ndarray: NaN-aware block averages along all three axes.

        Notes:
            This legacy helper uses ``self.nsub`` directly as one integer
            binning factor. A standard :class:`Nested3DGrid` stores one factor
            per level; :meth:`binning_onsubgrid_layered` is used internally
            with an explicit level-specific factor.
        """
        nbin = self.nsub
        d_avg = np.array([
            data[k::nbin, j::nbin, i::nbin]
            for k in range(nbin) for j in range(nbin) for i in range(nbin)
        ])
        return np.nanmean(d_avg, axis=0)

    def binning_onsubgrid_layered(
            self, data: np.ndarray, nbin: int
            ) -> np.ndarray | int:
        """Average refined spatial cells while preserving leading axes.

        Args:
            data (np.ndarray): Data with three spatial dimensions and zero,
                one, or two leading dimensions.
            nbin (int): Number of refined cells per parent cell along each
                spatial axis.

        Returns:
            np.ndarray or int: NaN-aware block averages. Integer ``0`` is
                returned when ``data`` does not have three to five dimensions.
        """
        dshape = len(data.shape)
        if dshape == 3:
            d_avg = np.array([data[k::nbin, j::nbin, i::nbin]
                              for k in range(nbin)
                              for j in range(nbin)
                              for i in range(nbin)])
        elif dshape == 4:
            d_avg = np.array([data[:, k::nbin, j::nbin, i::nbin]
                              for k in range(nbin)
                              for j in range(nbin)
                              for i in range(nbin)])
        elif dshape == 5:
            d_avg = np.array([data[:, :, k::nbin, j::nbin, i::nbin]
                              for k in range(nbin)
                              for j in range(nbin)
                              for i in range(nbin)])
        else:
            print('ERROR\tbinning_onsubgrid_layered: only Nd of data of 3-5 is now supported.')
            return 0
        return np.nanmean(d_avg, axis=0)

    def gridinfo(self, units: list[str] = ['au', 'au', 'au']) -> None:
        """Print the resolution and coordinate limits of every grid level.

        Args:
            units (list, optional): Labels for the x, y, and z coordinate
                units. Defaults to ``['au', 'au', 'au']``.
        """
        ux, uy, uz = units
        print('Nesting level: %i' % self.nlevels)
        print('Resolutions:')
        for l in range(self.nlevels):
            dx = self.xaxes[l][1] - self.xaxes[l][0]
            dy = self.yaxes[l][1] - self.yaxes[l][0]
            dz = self.zaxes[l][1] - self.zaxes[l][0]
            print('   l=%i: (dx, dy, dz) = (%.2e %s, %.2e %s, %.2e %s)' % (l, dx, ux, dy, uy, dz, uz))
            print('      : (xlim, ylim, zlim) = (%.2e to %.2e %s, %.2e to %.2e %s, %.2e to %.2e %s, )' % (
                self.xlim[l][0], self.xlim[l][1], ux,
                self.ylim[l][0], self.ylim[l][1], uy,
                self.zlim[l][0], self.zlim[l][1], uz))


def index_between(
        t: np.ndarray, tlim: list[float] | tuple[float, float],
        mode: str = 'all'
        ) -> np.ndarray | tuple[list[int], ...]:
    """Select values or index extents inside an inclusive interval.

    Args:
        t (np.ndarray): Coordinate array to examine.
        tlim (list or tuple): Inclusive lower and upper limits. A sequence
            whose length is not two indicates that no interval is applied.
        mode (str, optional): ``'all'`` returns a Boolean mask; ``'edge'``
            returns ``[minimum, maximum]`` index pairs for each dimension.
            Other values print a warning and fall back to the ``'all'``
            result. Defaults to ``'all'``.

    Returns:
        np.ndarray or tuple: Boolean selection mask in ``'all'`` mode, or one
            index pair per dimension in ``'edge'`` mode.

    Raises:
        ValueError: If ``mode='edge'`` and no element lies inside ``tlim``.
    """
    if not (len(tlim) == 2):
        if mode == 'all':
            return np.full(np.shape(t), True)
        elif mode == 'edge':
            if len(t.shape) == 1:
                return tuple([[0, len(t)-1]])
            else:
                return tuple([[0, t.shape[i]] for i in range(len(t.shape))])
        else:
            print('index_between: mode parameter is not right.')
            return np.full(np.shape(t), True)
    else:
        if mode == 'all':
            return (tlim[0] <= t) * (t <= tlim[1])
        elif mode == 'edge':
            nonzero = np.nonzero((tlim[0] <= t) * (t <= tlim[1]))
            return tuple([[np.min(i), np.max(i)] for i in nonzero])
        else:
            print('index_between: mode parameter is not right.')
            return (tlim[0] <= t) * (t <= tlim[1])


def nestgrid_3D(
        x: np.ndarray, y: np.ndarray, z: np.ndarray,
        xlim: list[float], ylim: list[float], zlim: list[float], nsub: int
        ) -> tuple[int, int, int, int, int, int,
                   np.ndarray, np.ndarray, np.ndarray] | int:
    """Refine a rectangular region of a three-dimensional Cartesian grid.

    Args:
        x (np.ndarray): One-dimensional parent-grid x coordinates.
        y (np.ndarray): One-dimensional parent-grid y coordinates.
        z (np.ndarray): One-dimensional parent-grid z coordinates.
        xlim (list): Inclusive minimum and maximum x coordinates to refine.
        ylim (list): Inclusive minimum and maximum y coordinates to refine.
        zlim (list): Inclusive minimum and maximum z coordinates to refine.
        nsub (int): Number of child cells placed across each selected parent
            cell along every axis.

    Returns:
        tuple or int: Inclusive parent-grid index limits ``ximin``, ``ximax``,
            ``yimin``, ``yimax``, ``zimin``, and ``zimax``, followed by the
            child-grid x, y, and z cell centers. Integer ``0`` is returned if
            any coordinate-limit sequence does not contain exactly two
            values.
    """
    # error check
    if (len(xlim) != 2) | (len(ylim) != 2) | (len(zlim) != 2):
        print('ERROR\tnest: Input xlim/ylim/zlim must be list as [min, max].')
        return 0

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    ximin, ximax = index_between(x, xlim, mode='edge')[0]  # starting and ending index of the subgrid
    yimin, yimax = index_between(y, ylim, mode='edge')[0]  # starting and ending index of the subgrid
    zimin, zimax = index_between(z, zlim, mode='edge')[0]  # starting and ending index of the subgrid
    _nx = ximax - ximin + 1
    _ny = yimax - yimin + 1
    _nz = zimax - zimin + 1
    xemin, xemax = x[ximin] - 0.5 * dx, x[ximax] + 0.5 * dx
    yemin, yemax = y[yimin] - 0.5 * dy, y[yimax] + 0.5 * dy
    zemin, zemax = z[zimin] - 0.5 * dz, z[zimax] + 0.5 * dz

    # nested grid
    xe_sub = np.linspace(xemin, xemax, _nx * nsub + 1)
    ye_sub = np.linspace(yemin, yemax, _ny * nsub + 1)
    ze_sub = np.linspace(zemin, zemax, _nz * nsub + 1)
    x_sub = 0.5 * (xe_sub[:-1] + xe_sub[1:])
    y_sub = 0.5 * (ye_sub[:-1] + ye_sub[1:])
    z_sub = 0.5 * (ze_sub[:-1] + ze_sub[1:])
    return ximin, ximax, yimin, yimax, zimin, zimax, x_sub, y_sub, z_sub
