# modules
import numpy as np
import math


class Nested3DGrid(object):
    """docstring for NestedGrid"""
    def __init__(self, x, y, z, 
        xlim = None, ylim = None, zlim = None, 
        nsub = None, reslim = 10,):
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
        nlevels = 1 if nsub is None else len(nsub) + 1
        self.nlevels = nlevels
        # original 1D axes
        self.xaxes = [None] * nlevels
        self.yaxes = [None] * nlevels
        self.zaxes = [None] * nlevels
        self.xaxes[0], self.yaxes[0], self.zaxes[0] = x, y, z
        # grid sizes
        self.ngrids = np.zeros((nlevels, 3)).astype(int)
        self.ngrids[0,:] = np.array([nx, ny, nz])
        # nested grid
        self.xnest = self.xx.ravel() # save all grid info in 1D array
        self.ynest = self.yy.ravel()
        self.znest = self.zz.ravel()
        self.dznest = np.full(self.znest.size, dz)
        self.partition = [0, self.xnest.size] # partition indices
        # starting and ending indices
        self.xinest = [-1,-1]
        self.yinest = [-1,-1]
        self.zinest = [-1,-1]
        # nest
        if self.nlevels > 1:
            if (np.array([xlim, ylim, zlim]) == None).any():
                _xlim, _ylim, _zlim = self.get_nestinglim(reslim = reslim)
                if xlim is None: xlim = _xlim
                if ylim is None: ylim = _ylim
                if zlim is None: zlim = _zlim
            self.xlim, self.ylim, self.zlim = xlim.copy(), ylim.copy(), zlim.copy()
            self.xlim.insert(0, [xe[0], xe[-1]])
            self.ylim.insert(0, [ye[0], ye[-1]])
            self.zlim.insert(0, [ze[0], ze[-1]])

            self.nest()
        else:
            self.xlim = [xe[0], xe[-1]]
            self.ylim = [ye[0], ye[-1]]
            self.zlim = [ze[0], ze[-1]]


    def get_nestinglim(self, reslim = 5):
        xlim = []
        ylim = []
        zlim = []
        _dx, _dy, _dz = np.abs(self.dx), np.abs(self.dy), np.abs(self.dz)
        for l in range(self.nlevels - 1):
            xlim.append([-_dx * reslim, _dx * reslim])
            ylim.append([-_dy * reslim, _dy * reslim])
            zlim.append([-_dz * reslim, _dz * reslim])
            _dx, _dy, _dz = np.abs(np.array([_dx, _dy, _dz])) / self.nsub[l]

        return xlim, ylim, zlim


    def get_grid(self, l):
        '''
        Get grid on the l layer.
        '''
        _nx, _ny, _nz = self.ngrids[l,:]
        partition = self.partition[l:l+2]
        # if it is not collapsed
        if self.xnest[partition[0]:partition[1]].size == _nx * _ny * _nz:
            xx = self.xnest[partition[0]:partition[1]].reshape(_nx, _ny, _nz)
            yy = self.ynest[partition[0]:partition[1]].reshape(_nx, _ny, _nz)
            zz = self.znest[partition[0]:partition[1]].reshape(_nx, _ny, _nz)
        else:
            # else
            x, y, z = self.xaxes[l], self.yaxes[l], self.zaxes[l]
            xx, yy, zz = np.meshgrid(x, y, z, indexing = 'ij')
        return xx, yy, zz


    def nest(self,):
        '''
        l - 1 is the mother grid layer. l is the child grid layer.
        '''
        # initialize
        partition = [0]
        xnest = np.array([])
        ynest = np.array([])
        znest = np.array([])
        dxnest = np.array([])
        dynest = np.array([])
        dznest = np.array([])
        for l in range(1, self.nlevels):
            # axes of the parental grid
            x, y, z = self.xaxes[l-1], self.yaxes[l-1], self.zaxes[l-1]
            dx = x[1] - x[0]
            dy = y[1] - y[0]
            dz = z[1] - z[0]

            # make childe grid
            ximin, ximax, yimin, yimax, zimin, zimax, x_sub, y_sub, z_sub = \
            nestgrid_3D(x, y, z, self.xlim[l], self.ylim[l], self.zlim[l], self.nsub[l-1])
            self.xinest += [ximin, ximax] # starting and ending indices on the upper-layer grid
            self.yinest += [yimin, yimax]
            self.zinest += [zimin, zimax]
            self.xaxes[l], self.yaxes[l], self.zaxes[l] = x_sub, y_sub, z_sub
            self.ngrids[l,:] = np.array([len(x_sub), len(y_sub), len(z_sub)])

            # parental grid
            _nx, _ny, _nz = self.ngrids[l-1,:]
            #if self.xnest[l-1].size == _nx * _ny * _nz:
            #    xx = self.xnest[l-1].reshape(_nx, _ny, _nz)
            #    yy = self.ynest[l-1].reshape(_nx, _ny, _nz)
            #    zz = self.znest[l-1].reshape(_nx, _ny, _nz)
            #else:
            xx, yy, zz = np.meshgrid(x, y, z, indexing = 'ij')

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

            Rx = np.concatenate([R1x, R2x, R3x, R4x, R5x, R6x])
            nl = Rx.size
            partition.append(partition[l-1] + nl)
            xnest = np.concatenate([xnest, Rx]) # update
            ynest = np.concatenate([ynest, R1y, R2y, R3y, R4y, R5y, R6y]) # update
            znest = np.concatenate([znest, R1z, R2z, R3z, R4z, R5z, R6z]) # update
            dxnest = np.concatenate([dxnest, np.full(nl, dx)])
            dynest = np.concatenate([dynest, np.full(nl, dy)])
            dznest = np.concatenate([dznest, np.full(nl, dz)])

        # the deepest child grid
        xx_sub, yy_sub, zz_sub = np.meshgrid(x_sub, y_sub, z_sub, indexing = 'ij')
        xnest = np.concatenate([xnest, xx_sub.ravel()]) # update
        ynest = np.concatenate([ynest, yy_sub.ravel()]) # update
        znest = np.concatenate([znest, zz_sub.ravel()]) # update

        nl = xx_sub.size
        dx = x_sub[1] - x_sub[0]
        dy = y_sub[1] - y_sub[0]
        dz = z_sub[1] - z_sub[0]
        dxnest = np.concatenate([dxnest, np.full(nl, dx)])
        dynest = np.concatenate([dynest, np.full(nl, dy)])
        dznest = np.concatenate([dznest, np.full(nl, dz)])

        nd = xnest.size
        partition.append(nd)

        self.xnest = xnest
        self.ynest = ynest
        self.znest = znest
        self.dxnest = dxnest
        self.dynest = dynest
        self.dznest = dznest
        self.partition = partition
        self.nd = nd


    def collapse(self, d, upto = None):
        '''
        Collapse given data to the mother grid.

        Parameters
        ----------
        d (list): List of data on the nested grid
        '''
        lmax = 0 if upto is None else upto
        d_col = d[self.partition[-2]:self.partition[-1]] # starting from the inner most grid
        d_col = d_col.reshape(tuple(self.ngrids[-1,:]))
        for l in range(self.nlevels-1,lmax,-1):
            nsub = self.nsub[l-1]
            ximin, ximax = self.xinest[l*2:(l+1)*2]
            yimin, yimax = self.yinest[l*2:(l+1)*2]
            zimin, zimax = self.zinest[l*2:(l+1)*2]
            # collapse data on the inner grid
            _d = self.binning_onsubgrid_layered(d_col, nsub)
            #print(ximin, ximax, yimin, yimax, zimin, zimax)

            # go upper layer
            nx, ny, nz = self.ngrids[l-1,:] # size of the upper layer
            d_col = np.full((nx, ny, nz), np.nan)

            # insert collapsed data
            d_col[ximin:ximax+1, yimin:yimax+1, zimin:zimax+1] = _d

            # fill upper layer data
            d_up = d[self.partition[l-1]:self.partition[l]]
            # Region 1: x from zero to ximin, all y and z
            d_col[:ximin, :, :] = \
            d_up[:ximin * ny * nz].reshape((ximin, ny, nz))
            # Region 2: x from ximax to nx, all y and z
            i0 = ximin * ny * nz
            i1 = i0 + (nx - ximax - 1) * ny * nz
            d_col[ximax+1:, :, :] = \
            d_up[i0:i1].reshape(
                (nx - ximax - 1, ny, nz))
            # Region 3
            i0 = i1
            i1 = i0 + (ximax + 1 - ximin) * yimin * nz
            d_col[ximin:ximax+1, :yimin, :] = \
            d_up[i0:i1].reshape(
                (ximax + 1 - ximin, yimin, nz))
            # Region 4
            i0 = i1
            i1 = i0 + (ximax + 1 - ximin) * (ny - yimax - 1) * nz
            d_col[ximin:ximax+1, yimax+1:, :] = \
            d_up[i0:i1].reshape(
                (ximax + 1 - ximin, ny - yimax - 1, nz))
            # Region 5
            i0 = i1
            i1 = i0 + (ximax + 1 - ximin) * (yimax + 1 - yimin) * zimin
            d_col[ximin:ximax+1, yimin:yimax+1, :zimin] = \
            d_up[i0:i1].reshape(
                (ximax + 1 - ximin, yimax + 1 - yimin, zimin))
            # Region 6
            i0 = i1
            i1 = i0 + (ximax + 1 - ximin) * (yimax + 1 - yimin) * (nz - zimax -1)
            d_col[ximin:ximax+1, yimin:yimax+1, zimax+1:] = \
            d_up[i0:].reshape(
                (ximax + 1 - ximin, yimax + 1 - yimin, nz - zimax -1))

        return d_col


    def high_dimensional_collapse(self, d, upto = None, fill = 'nan'):
        if len(d.shape) == 2:
            d_col = self.collapse_extra_1d(d, upto = upto, fill = fill)
            return d_col
        else:
            print('ERROR\thigh_dimensional_collapse: currently only 2d data are supported.')
            return 0


    def collapse_extra_1d(self, d, upto = None, fill = 'nan'):
        '''
        Collapse given data to the mother grid.

        Parameters
        ----------
        d (list): List of data on the nested grid
        '''
        lmax = 0 if upto is None else upto
        nv, nd = d.shape
        d_col = d[:, self.partition[-2]:self.partition[-1]] # starting from the inner most grid
        d_col = d_col.reshape((nv, self.ngrids[-1,:][0], self.ngrids[-1,:][1], self.ngrids[-1,:][2]))
        for l in range(self.nlevels-1,lmax,-1):
            nsub = self.nsub[l-1]
            ximin, ximax = self.xinest[l*2:(l+1)*2]
            yimin, yimax = self.yinest[l*2:(l+1)*2]
            zimin, zimax = self.zinest[l*2:(l+1)*2]
            # collapse data on the inner grid
            _d = self.binning_onsubgrid_layered(d_col, nsub)
            #print(ximin, ximax, yimin, yimax, zimin, zimax)

            # go upper layer
            nx, ny, nz = self.ngrids[l-1,:] # size of the upper layer
            if fill == 'nan':
                d_col = np.full((nv, nx, ny, nz), np.nan)
            elif fill == 'zero':
                d_col = np.zeros((nv, nx, ny, nz))
            else:
                d_col = np.full((nv, nx, ny, nz), fill)

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

        return d_col


    def nest_sub(self, xlim,  ylim, zlim, nsub):
        # error check
        if (len(xlim) != 2) | (len(ylim) != 2) | (len(zlim) != 2):
            print('ERROR\tnest: Input xlim/ylim/zlim must be list as [min, max].')
            return 0
        # decimals
        xlim = [np.round(xlim[0], self.decimals), np.round(xlim[1], self.decimals)]
        ylim = [np.round(ylim[0], self.decimals), np.round(ylim[1], self.decimals)]

        self.nsub = nsub
        self.xlim_sub, self.ylim_sub, self.zlim_sub = xlim, ylim, zlim
        ximin, ximax = index_between(self.x, xlim, mode='edge')[0]
        yimin, yimax = index_between(self.y, ylim, mode='edge')[0]
        zimin, zimax = index_between(self.z, zlim, mode='edge')[0]
        _nx = ximax - ximin + 1
        _ny = yimax - yimin + 1
        _nz - zimax - zimin + 1
        xemin, xemax = self.xe[ximin], self.xe[ximax + 1]
        yemin, yemax = self.ye[yimin], self.ye[yimax + 1]
        zemin, zemax = self.ze[zimin], self.ze[zimax + 1]
        self.xi0, self.xi1 = ximin, ximax # Starting and ending indices of nested grid
        self.yi0, self.yi1 = yimin, yimax # Starting and ending indices of nested grid
        self.zi0, self.zi1 = zimin, zimax # Starting and ending indices of nested grid

        # nested grid
        xe_sub = np.linspace(xemin, xemax, _nx * nsub + 1)
        ye_sub = np.linspace(yemin, yemax, _ny * nsub + 1)
        ze_sub = np.linspace(zemin, zemax, _nz * nsub + 1)
        x_sub = 0.5 * (xe_sub[:-1] + xe_sub[1:])
        y_sub = 0.5 * (ye_sub[:-1] + ye_sub[1:])
        z_sub = 0.5 * (ze_sub[:-1] + ze_sub[1:])
        xx_sub, yy_sub, zz_sub = np.meshgrid(x_sub, y_sub, z_sub, indexing = 'ij')
        self.xe_sub, self.ye_sub, self.ze_sub = xe_sub, ye_sub, ze_sub
        self.x_sub, self.y_sub, z_sub = x_sub, y_sub, z_sub
        self.xx_sub, self.yy_sub, self.zz_sub = xx_sub, yy_sub, zz_sub
        self.dx_sub, self.dy_sub, self.dz_sub = self.dx / nsub, self.dy / nsub, self.dz / nsub
        self.nx_sub, self.ny_sub, self.nz_sub = len(x_sub), len(y_sub), len(z_sub)
        return xx_sub, yy_sub, zz_sub


    def where_subgrid(self):
        return np.where(
            (self.xx >= self.xlim_sub[0]) * (self.xx <= self.xlim_sub[1]) \
            * (self.yy >= self.ylim_sub[0]) * (self.yy <= self.ylim_sub[1]))


    def binning_onsubgrid(self, data):
        nbin = self.nsub
        d_avg = np.array([
            data[k::nbin, j::nbin, i::nbin]
            for k in range(nbin) for j in range(nbin) for i in range(nbin)
            ])
        return np.nanmean(d_avg, axis = 0)


    def binning_onsubgrid_layered(self, data, nbin):
        dshape = len(data.shape)
        if dshape == 3:
            d_avg = np.array([
                data[k::nbin, j::nbin, i::nbin]
                for k in range(nbin) for j in range(nbin) for i in range(nbin)
                ])
        elif dshape == 4:
            d_avg = np.array([
                data[:, k::nbin, j::nbin, i::nbin]
                for k in range(nbin) for j in range(nbin) for i in range(nbin)
                ])
        elif dshape ==5:
            d_avg = np.array([
                data[:, :, k::nbin, j::nbin, i::nbin]
                for k in range(nbin) for j in range(nbin) for i in range(nbin)
                ])
        else:
            print('ERROR\tbinning_onsubgrid_layered: only Nd of data of 3-5 is now supported.')
            return 0
        return np.nanmean(d_avg, axis = 0)


    def integrate_along(self, d, axis = 'z'):
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


    def integrate_along_z(self, d, fill = 'nan'):
        '''
        Collapse given data to the mother grid.

        Parameters
        ----------
        d (list): List of data on the nested grid
        '''
        lmax = 0
        nv, nd = d.shape
        d_col = d[:, self.partition[-2]:self.partition[-1]] # starting from the inner most grid
        d_col = d_col.reshape((nv, self.ngrids[-1,:][0], self.ngrids[-1,:][1], self.ngrids[-1,:][2]))

        # dz of the inner most grid
        z = self.zaxes[-1]
        dz = z[1] - z[0]

        for l in range(self.nlevels-1,lmax,-1):
            nsub = self.nsub[l-1]
            ximin, ximax = self.xinest[l*2:(l+1)*2]
            yimin, yimax = self.yinest[l*2:(l+1)*2]
            zimin, zimax = self.zinest[l*2:(l+1)*2]
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

        return np.nansum(d_col * dz, axis = 3)


    def binning_z_integrated(self, data, nbin):
        dshape = len(data.shape)
        if dshape == 3:
            d_avg = np.array([[
                data[k::nbin, j::nbin, i::nbin]
                for k in range(nbin) for j in range(nbin)]
                for i in range(nbin)
                ])
        elif dshape == 4:
            d_avg = np.array([[
                data[:, k::nbin, j::nbin, i::nbin]
                for k in range(nbin) for j in range(nbin)]
                for i in range(nbin)
                ])
        elif dshape ==5:
            d_avg = np.array([[
                data[:, :, k::nbin, j::nbin, i::nbin]
                for k in range(nbin) for j in range(nbin)] 
                for i in range(nbin)
                ])
        else:
            print('ERROR\tbinning_onsubgrid_layered: only Nd of data of 3-5 is now supported.')
            return 0

        # integrate & binning
        d_avg = np.nansum(d_avg, axis = 0) # integrate
        d_avg = np.nanmean(d_avg, axis = 0) # binning
        return d_avg



    def gridinfo(self, units = ['au', 'au', 'au']):
        ux, uy, uz = units
        print('Nesting level: %i'%self.nlevels)
        print('Resolutions:')
        for l in range(self.nlevels):
            dx = self.xaxes[l][1] - self.xaxes[l][0]
            dy = self.yaxes[l][1] - self.yaxes[l][0]
            dz = self.zaxes[l][1] - self.zaxes[l][0]
            print('   l=%i: (dx, dy, dz) = (%.2e %s, %.2e %s, %.2e %s)'%(l, dx, ux, dy, uy, dz, uz))
            print('      : (xlim, ylim, zlim) = (%.2e to %.2e %s, %.2e to %.2e %s, %.2e to %.2e %s, )'%(
                self.xlim[l][0], self.xlim[l][1], ux,
                self.ylim[l][0], self.ylim[l][1], uy,
                self.zlim[l][0], self.zlim[l][1], uz))


    def edgecut_indices(self, xlength, ylength):
        # odd or even
        x_oddeven = self.nx%2
        y_oddeven = self.ny%2
        # edge indices for subgrid
        xi = int(xlength / self.dx_sub) if self.dx_sub > 0 else int(- xlength / self.dx_sub)
        yi = int(ylength / self.dy_sub)
        _nx_resub = int(self.nx_sub - 2 * xi) // self.nsub # nx of subgrid after cutting edge
        _ny_resub = int(self.ny_sub - 2 * yi) // self.nsub # ny of subgrid after cutting edge
        # fit odd/even
        if _nx_resub%2 != x_oddeven: _nx_resub += 1
        if _ny_resub%2 != y_oddeven: _ny_resub += 1
        # nx, ny of the new subgrid and new xi and yi
        nx_resub = _nx_resub * self.nsub
        ny_resub = _ny_resub * self.nsub
        xi = (self.nx_sub - nx_resub) // 2
        yi = (self.ny_sub - ny_resub) // 2
        # for original grid
        xi0 = int((self.nx - nx_resub / self.nsub) * 0.5)
        yi0 = int((self.ny - ny_resub / self.nsub) * 0.5)
        #print(nx_resub / self.nsub, self.nx - xi0 - xi0)
        return xi, yi, xi0, yi0


def index_between(t, tlim, mode='all'):
    if not (len(tlim) == 2):
        if mode=='all':
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
        if mode=='all':
            return (tlim[0] <= t) * (t <= tlim[1])
        elif mode == 'edge':
            nonzero = np.nonzero((tlim[0] <= t) * (t <= tlim[1]))
            return tuple([[np.min(i), np.max(i)] for i in nonzero])
        else:
            print('index_between: mode parameter is not right.')
            return (tlim[0] <= t) * (t <= tlim[1])


def nestgrid_3D(x, y, z, xlim, ylim, zlim, nsub, decimals = 4.):
    # error check
    if (len(xlim) != 2) | (len(ylim) != 2) | (len(zlim) != 2):
        print('ERROR\tnest: Input xlim/ylim/zlim must be list as [min, max].')
        return 0
    # decimals
    #xlim = [np.round(xlim[0], self.decimals), np.round(xlim[1], self.decimals)]
    #ylim = [np.round(ylim[0], self.decimals), np.round(ylim[1], self.decimals)]

    dx = x[1] - x[0]
    dy = y[1] - y[0]
    dz = z[1] - z[0]
    ximin, ximax = index_between(x, xlim, mode='edge')[0] # starting and ending index of the subgrid
    yimin, yimax = index_between(y, ylim, mode='edge')[0] # starting and ending index of the subgrid
    zimin, zimax = index_between(z, zlim, mode='edge')[0] # starting and ending index of the subgrid
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