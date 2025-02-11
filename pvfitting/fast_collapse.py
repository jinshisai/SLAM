import numpy as np
from numba import jit
from numba import prange


@jit(parallel=True)
def fast_integration_along(self, d, axis = 'z'):
    shape = d.shape
    ndim = len(shape)
    if ndim == 1:
        _d = d.reshape((1, ndim))
        outshape = (self.nx, self.ny)
    elif ndim >= 2:
        _n = math.prod(list(shape)[:-1])
        _d = d.reshape((_n, shape[-1]))
        outshape = tuple(list(shape[:-1]) + [self.nx, self.ny])

    if axis == 'z':
        _d = self.integrate_along_z(_d)
    else:
        print('ERROR\tintegrate_along: integration along z-axis is only supported for now.')
        return 0

    return _d.reshape(outshape)


@jit(parallel=True)
def fast_integration_along_z(self, 
    d, grid,):
    '''
    Collapse given data to the mother grid.

    Parameters
    ----------
    d (list): List of data on the nested grid
    '''
    # grid info
    nlevels = grid.nlevels
    nsub = grid.nsub
    dzs = grid.dzs
    partition = grid.partition
    ngrids = grid.ngrids
    nv, nd = d.shape
    xinest = grid.xinest
    yinest = grid.yinest
    zinest = grid.zinest

    d_col = d[:, partition[-2]:partition[-1]] # starting from the inner most grid
    d_col = d_col.reshape((nv, ngrids[-1,:][0], ngrids[-1,:][1], ngrids[-1,:][2]))

    # dz of the inner most grid
    dz = dzs[-1]

    for ch in prange(nv):
        for l in range(nlevels):
            nsub = nsub[l-1]
            ximin, ximax = xinest[l*2:(l+1)*2]
            yimin, yimax = yinest[l*2:(l+1)*2]
            zimin, zimax = zinest[l*2:(l+1)*2]
            # collapse data on the inner grid
            _d = self.binning_z_integrated(d_col, nsub) # integration along z within each parental cell
            _d *= dz

        # go upper layer
        nx, ny, nz = self.ngrids[l-1,:] # size of the upper layer
        d_col = np.full((nv, nx, ny, nz), np.nan)

        # insert collapsed data
        d_col[:, ximin:ximax+1, yimin:yimax+1, zimin:zimax+1] = _d

        # fill upper layer data
        d_up = d[:, self.partition[l-1]:self.partition[l]]
        # Region 1: x from zero to ximin, all y and z
        d_col[:, :ximin, :, :] = \
        d_up[:, :ximin * ny * nz].reshape((nv, ximin, ny, nz))
        # Region 2: x from ximax to nx, all y and z
        i0 = ximin * ny * nz
        i1 = i0 + (nx - ximax - 1) * ny * nz
        d_col[:, ximax+1:, :, :] = \
        d_up[:, i0:i1].reshape(
            (nv, nx - ximax - 1, ny, nz))
        # Region 3
        i0 = i1
        i1 = i0 + (ximax + 1 - ximin) * yimin * nz
        d_col[:, ximin:ximax+1, :yimin, :] = \
        d_up[:, i0:i1].reshape(
            (nv, ximax + 1 - ximin, yimin, nz))
        # Region 4
        i0 = i1
        i1 = i0 + (ximax + 1 - ximin) * (ny - yimax - 1) * nz
        d_col[:, ximin:ximax+1, yimax+1:, :] = \
        d_up[:, i0:i1].reshape(
            (nv, ximax + 1 - ximin, ny - yimax - 1, nz))
        # Region 5
        i0 = i1
        i1 = i0 + (ximax + 1 - ximin) * (yimax + 1 - yimin) * zimin
        d_col[:, ximin:ximax+1, yimin:yimax+1, :zimin] = \
        d_up[:, i0:i1].reshape(
            (nv, ximax + 1 - ximin, yimax + 1 - yimin, zimin))
        # Region 6
        i0 = i1
        i1 = i0 + (ximax + 1 - ximin) * (yimax + 1 - yimin) * (nz - zimax -1)
        d_col[:, ximin:ximax+1, yimin:yimax+1, zimax+1:] = \
        d_up[:, i0:].reshape(
            (nv, ximax + 1 - ximin, yimax + 1 - yimin, nz - zimax -1))

        # update dz axis
        z = self.zaxes[l-1] # parental axis
        dz = z[1] - z[0]

    d_col *= dz
    return np.nansum(d_col, axis = 3)


@jit(parallel=True)
def fast_binning_z_integrated(data, nbin, dz):
    nv, nx, ny, nz = data.shape
    nx_new, ny_new, nz_new = nx//nbin, ny//nbin, nz//nbin
    d_avg = np.zeros((nv, nx_new, ny_new, nz_new))

    for ch in prange(nv):
        for i in range(nx_new):
            for j in range(ny_new):
                for k in range(nz_new):
                    isub = range(i*nbin, i*nbin + nbin)
                    jsub = range(j*nbin, j*nbin + nbin)
                    ksub = range(k*nbin, k*nbin + nbin)

                    # integration along z and binning in xy-plane
                    d_binned = fast_binning(data[ch,:,:,:], isub, jsub, ksub)
                    d_binned /= nbin * nbin # mean in xy-plane
                    d_avg[ch, i, j, k] = d_binned

    return d_avg



@jit(parallel=False)
def fast_binning(d, xsubrange, ysubrange, zsubrange):
    d_sum = 0.
    for i in xsubrange:
        for j in ysubrange:
            for k in zsubrange:
                d_sum += d[i,j,k]
    return d_sum
