# modules
import numpy as np
import matplotlib.pyplot as plt



class Nested3DGrid(object):
    """docstring for NestedGrid"""
    def __init__(self, x, y, z, 
        xlim, ylim, zlim, nsub, 
        nlevels = 1, reslim = 5,):
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
            if (np.array([xlim, ylim, zlim]) == None).any():
                _xlim, _ylim, _zlim = self.get_nestinglim(reslim = reslim)
                if xlim is None: xlim = _xlim
                if ylim is None: ylim = _ylim
                if zlim is None: zlim = _zlim
            for l in range(nlevels):
                self.nest(l+1, xlim[l], ylim[l], zlim[l], nsub[l])
            self.xlim, self.ylim, self.zlim = xlim.copy(), ylim.copy(), zlim.copy()
            self.xlim.insert(0, [xe[0], xe[-1]])
            self.ylim.insert(0, [ye[0], ye[-1]])
            self.zlim.insert(0, [ze[0], ze[-1]])
        else:
            self.xlim = [xe[0], xe[-1]]
            self.ylim = [ye[0], ye[-1]]
            self.zlim = [ze[0], ze[-1]]


        '''
        if (_check := self.check_symmetry(precision))[0]:
            pass
        else:
            print('ERROR\tNested2DGrid: Input grid must be symmetric but not.')
            print('ERROR\tNested2DGrid: Condition.')
            print('ERROR\tNested2DGrid: [xcent, ycent, dx, dy]')
            print('ERROR\tNested2DGrid: ', _check[1])
            return None
        '''


    def get_nestinglim(self, reslim = 5):
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


    def check_symmetry(self, decimals = 5):
        nx, ny = self.nx, self.ny
        xc = np.round(self.xc, decimals)
        yc = np.round(self.yc, decimals)
        _xcent = (xc == 0.) if nx%2 == 1 else (xc == - np.round(self.xx[ny//2 - 1, nx//2 - 1], decimals))
        _ycent = (yc == 0.) if ny%2 == 1 else (yc == - np.round(self.yy[ny//2 - 1, nx//2 - 1], decimals))
        delxs = (self.xx[1:,1:] - self.xx[:-1,:-1]) / self.dx
        delys = (self.yy[1:,1:] - self.yy[:-1,:-1]) / self.dy
        _xdel = (np.round(delxs, decimals) == 1. ).all()
        _ydel = (np.round(delys, decimals)  == 1. ).all()
        cond = [_xdel, _ydel] # _xcent, _ycent,
        return all(cond), cond


    def get_grid(self, l):
        '''
        Get grid on the l layer.
        '''
        _nx, _ny, _nz = self.ngrids[l]
        # if it is not collapsed
        if self.xnest[l].size == _nx * _ny * _nz:
            xx = self.xnest[l].reshape(_nx, _ny, _nz)
            yy = self.ynest[l].reshape(_nx, _ny, _nz)
            zz = self.znest[l].reshape(_nx, _ny, _nz)
        else:
            # else
            x, y, z = self.xaxes[l], self.yaxes[l], self.zaxes[l]
            xx, yy, zz = np.meshgrid(x, y, z, indexing = 'ij')
        return xx, yy, zz


    def nest(self, l, xlim, ylim, zlim, nsub):
        '''
        l - 1 is the mother grid layer. l is the child grid layer.
        '''
        x, y, z = self.xaxes[l-1], self.yaxes[l-1], self.zaxes[l-1]
        ximin, ximax, yimin, yimax, zimin, zimax, x_sub, y_sub, z_sub = \
        nestgrid_3D(x, y, z, xlim, ylim, zlim, nsub)
        self.xinest[l] = [ximin, ximax] # starting and ending indices on the upper-layer grid
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
        self.xnest[l-1] = np.concatenate([R1x, R2x, R3x, R4x, R5x, R6x]) # update
        self.ynest[l-1] = np.concatenate([R1y, R2y, R3y, R4y, R5y, R6y]) # update
        self.znest[l-1] = np.concatenate([R1z, R2z, R3z, R4z, R5z, R6z]) # update

        # child grid
        xx_sub, yy_sub, zz_sub = np.meshgrid(x_sub, y_sub, z_sub, indexing = 'ij')
        self.xnest[l] = xx_sub.ravel()
        self.ynest[l] = yy_sub.ravel()
        self.znest[l] = zz_sub.ravel()
        self.ngrids[l] = (len(x_sub), len(y_sub), len(z_sub))


    def collapse(self, d, upto = None):
        '''
        Collapse given data to the mother grid.

        Parameters
        ----------
        d (list): List of data on the nested grid
        '''
        d_col = d[-1] # starting from the inner most grid
        lmax = 0 if upto is None else upto
        for l in range(self.nlevels-1,lmax,-1):
            nsub = self.nsub[l-1]
            ximin, ximax = self.xinest[l]
            yimin, yimax = self.yinest[l]
            zimin, zimax = self.zinest[l]
            # collapse data on the inner grid
            _d = self.binning_onsubgrid_layered(d_col.reshape(self.ngrids[l]), nsub)
            #print(ximin, ximax, yimin, yimax, zimin, zimax)

            # go next layer
            nx, ny, nz = self.ngrids[l-1] # size of the upper layer
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
            i1 = i0 + (ximax + 1 - ximin) * (yimax + 1 - yimin) * (nz - zimax -1)
            d_col[ximin:ximax+1, yimin:yimax+1, zimax+1:] = \
            d[l-1][i0:].reshape(
                (ximax + 1 - ximin, yimax + 1 - yimin, nz - zimax -1))

            #print(l)
            #print(np.nonzero(np.isnan(d_col)))

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
            data[i::nbin, i::nbin, i::nbin]
            for i in range(nbin)
            ])
        return np.nanmean(d_avg, axis = 0)


    def binning_onsubgrid_layered(self, data, nbin):
        dshape = len(data.shape)
        if dshape == 3:
            d_avg = np.array([
                data[i::nbin, i::nbin, i::nbin]
                for i in range(nbin)
                ])
        elif dshape == 4:
            d_avg = np.array([
                data[:, i::nbin, i::nbin, i::nbin]
                for i in range(nbin)
                ])
        elif dshape ==5:
            d_avg = np.array([
                data[:, :, i::nbin, i::nbin, i::nbin]
                for i in range(nbin)
                ])
        else:
            print('ERROR\tbinning_onsubgrid_layered: only Nd of data of 3-5 is now supported.')
            return 0
        return np.nanmean(d_avg, axis = 0)



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



class Nested2DGrid(object):
    """docstring for NestedGrid"""
    def __init__(self, xx, yy, precision = 4):
        super(Nested2DGrid, self).__init__()
        self.xx = xx
        self.yy = yy
        ny, nx = xx.shape
        self.ny, self.nx = ny, nx
        self.dx = xx[0,1] - xx[0,0]
        self.dy = yy[1,0] - yy[0,0]
        self.xc = xx[ny//2, nx//2]
        self.yc = yy[ny//2, nx//2]
        self.yci, self.xci = ny//2, nx//2

        if (_check := self.check_symmetry(precision))[0]:
            pass
        else:
            print('ERROR\tNested2DGrid: Input grid must be symmetric but not.')
            print('ERROR\tNested2DGrid: Condition.')
            print('ERROR\tNested2DGrid: [xcent, ycent, dx, dy]')
            print('ERROR\tNested2DGrid: ', _check[1])
            return None

        # retrive x and y
        x = self.xx[0,:]
        y = self.yy[:,0]
        xe = np.hstack([x - self.dx * 0.5, x[-1] + self.dx * 0.5])
        ye = np.hstack([y - self.dy * 0.5, y[-1] + self.dy * 0.5])
        self.x, self.y = x, y
        self.xe, self.ye = xe, ye
        self.decimals = precision


    def check_symmetry(self, decimals = 5):
        nx, ny = self.nx, self.ny
        xc = np.round(self.xc, decimals)
        yc = np.round(self.yc, decimals)
        _xcent = (xc == 0.) if nx%2 == 1 else (xc == - np.round(self.xx[ny//2 - 1, nx//2 - 1], decimals))
        _ycent = (yc == 0.) if ny%2 == 1 else (yc == - np.round(self.yy[ny//2 - 1, nx//2 - 1], decimals))
        delxs = (self.xx[1:,1:] - self.xx[:-1,:-1]) / self.dx
        delys = (self.yy[1:,1:] - self.yy[:-1,:-1]) / self.dy
        _xdel = (np.round(delxs, decimals) == 1. ).all()
        _ydel = (np.round(delys, decimals)  == 1. ).all()
        cond = [_xdel, _ydel] # _xcent, _ycent,
        return all(cond), cond
    

    def nest(self, xlim,  ylim, nsub = 2):
        # error check
        if (len(xlim) != 2) | (len(ylim) != 2):
            print('ERROR\tnest: Input xlim and/or ylim is not valid.')
            return 0
        # decimals
        xlim = [np.round(xlim[0], self.decimals), np.round(xlim[1], self.decimals)]
        ylim = [np.round(ylim[0], self.decimals), np.round(ylim[1], self.decimals)]

        self.nsub = nsub
        self.xlim_sub, self.ylim_sub = xlim, ylim
        ximin, ximax = index_between(self.x, xlim, mode='edge')[0]
        yimin, yimax = index_between(self.y, ylim, mode='edge')[0]
        _nx = ximax - ximin + 1
        _ny = yimax - yimin + 1
        xemin, xemax = self.xe[ximin], self.xe[ximax + 1]
        yemin, yemax = self.ye[yimin], self.ye[yimax + 1]
        self.xi0, self.xi1 = ximin, ximax # Starting and ending indices of nested grid
        self.yi0, self.yi1 = yimin, yimax # Starting and ending indices of nested grid

        # nested grid
        xe_sub = np.linspace(xemin, xemax, _nx * nsub + 1)
        ye_sub = np.linspace(yemin, yemax, _ny * nsub + 1)
        x_sub = 0.5 * (xe_sub[:-1] + xe_sub[1:])
        y_sub = 0.5 * (ye_sub[:-1] + ye_sub[1:])
        xx_sub, yy_sub = np.meshgrid(x_sub, y_sub)
        self.xe_sub, self.ye_sub = xe_sub, ye_sub
        self.x_sub, self.y_sub = x_sub, y_sub
        self.xx_sub, self.yy_sub = xx_sub, yy_sub
        self.dx_sub, self.dy_sub = self.dx / nsub, self.dy / nsub
        self.nx_sub, self.ny_sub = len(x_sub), len(y_sub)
        return xx_sub, yy_sub


    def where_subgrid(self):
        return np.where(
            (self.xx >= self.xlim_sub[0]) * (self.xx <= self.xlim_sub[1]) \
            * (self.yy >= self.ylim_sub[0]) * (self.yy <= self.ylim_sub[1]))


    def binning_onsubgrid(self, data):
        nbin = self.nsub
        d_avg = np.array([
            data[i::nbin, i::nbin]
            for i in range(nbin)
            ])
        return np.nanmean(d_avg, axis = 0)


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


    def binning(self, nbin):
        if nbin%2 == 0:
            xx, yy = self.shift()
        else:
            xx, yy = self.xx.copy(), self.yy.copy()

        xcut = self.nx%nbin
        ycut = self.ny%nbin
        _xx = xx[ycut//2:-ycut//2, xcut//2:-xcut//2]
        _yy = yy[ycut//2:-ycut//2, xcut//2:-xcut//2]
        xx_avg = np.array([
            _xx[i::nbin, i::nbin]
            for i in range(nbin)
            ])
        yy_avg = np.array([
            _yy[i::nbin, i::nbin]
            for i in range(nbin)
            ])

        return np.average(xx_avg, axis= 0), np.average(yy_avg, axis= 0)


    def shift(self):
        rex = np.arange(-self.nx//2, self.nx//2+1, 1) + 0.5
        rex *= self.dx
        rey = np.arange(-self.ny//2, self.ny//2+1, 1) + 0.5
        rey *= self.dy
        return np.meshgrid(rex, rey)


class SubGrid2D(object):
    """docstring for NestedGrid"""
    def __init__(self, x, y, nsub = 2):
        super(SubGrid2D, self).__init__()
        # retrive x and y
        self.x = x
        self.y = y
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        self.dx = dx
        self.dy = dy
        xe = np.hstack([x - self.dx * 0.5, x[-1] + self.dx * 0.5])
        ye = np.hstack([y - self.dy * 0.5, y[-1] + self.dy * 0.5])
        self.xe, self.ye = xe, ye

        # save grid info
        #xx, yy = np.meshgrid(x,y)
        #self.xx = xx
        #self.yy = yy
        ny, nx = len(y), len(x)
        self.ny, self.nx = ny, nx

        # subgrid
        self.subgrid(nsub = nsub)


    def subgrid(self, nsub = 2):
        self.nsub = nsub
        nx_sub, ny_sub = self.nx * nsub, self.ny * nsub
        self.nx_sub, self.ny_sub = nx_sub, ny_sub

        # sub grid
        xemin, xemax = self.xe[0], self.xe[-1] # edge of the original grid
        yemin, yemax = self.ye[0], self.ye[-1] # edge of the original grid
        xe_sub = np.linspace(xemin, xemax, nx_sub + 1)
        ye_sub = np.linspace(yemin, yemax, ny_sub + 1)
        x_sub = 0.5 * (xe_sub[:-1] + xe_sub[1:])
        y_sub = 0.5 * (ye_sub[:-1] + ye_sub[1:])
        xx_sub, yy_sub = np.meshgrid(x_sub, y_sub)
        self.xe_sub, self.ye_sub = xe_sub, ye_sub
        self.x_sub, self.y_sub = x_sub, y_sub
        self.xx_sub, self.yy_sub = xx_sub, yy_sub
        self.dx_sub, self.dy_sub = self.dx / nsub, self.dy / nsub
        #self.nx_sub, self.ny_sub = len(x_sub), len(y_sub)
        return xx_sub, yy_sub


    def binning_onsubgrid(self, data):
        nbin = self.nsub
        d_avg = np.array([
            data[i::nbin, i::nbin]
            for i in range(nbin)
            ])
        return np.nanmean(d_avg, axis = 0)


    def binning_onsubgrid_layered(self, data):
        nbin = self.nsub
        dshape = len(data.shape)
        if dshape == 2:
            d_avg = np.array([
                data[i::nbin, i::nbin]
                for i in range(nbin)
                ])
        elif dshape == 3:
            d_avg = np.array([
                data[:, i::nbin, i::nbin]
                for i in range(nbin)
                ])
        elif dshape ==4:
            d_avg = np.array([
                data[:, :, i::nbin, i::nbin]
                for i in range(nbin)
                ])
        else:
            print('ERROR\tbinning_onsubgrid_layered: only Nd of data of 2-4 is now supported.')
            return 0
        return np.nanmean(d_avg, axis = 0)


    def binning(self, nbin):
        if nbin%2 == 0:
            xx, yy = self.shift()
        else:
            xx, yy = self.xx.copy(), self.yy.copy()

        xcut = self.nx%nbin
        ycut = self.ny%nbin
        _xx = xx[ycut//2:-ycut//2, xcut//2:-xcut//2]
        _yy = yy[ycut//2:-ycut//2, xcut//2:-xcut//2]
        xx_avg = np.array([
            _xx[i::nbin, i::nbin]
            for i in range(nbin)
            ])
        yy_avg = np.array([
            _yy[i::nbin, i::nbin]
            for i in range(nbin)
            ])

        return np.average(xx_avg, axis= 0), np.average(yy_avg, axis= 0)


    def shift(self):
        rex = np.arange(-self.nx//2, self.nx//2+1, 1) + 0.5
        rex *= self.dx
        rey = np.arange(-self.ny//2, self.ny//2+1, 1) + 0.5
        rey *= self.dy
        return np.meshgrid(rex, rey)


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
    xx_sub, yy_sub, zz_sub = np.meshgrid(x_sub, y_sub, z_sub, indexing = 'ij')
    return ximin, ximax, yimin, yimax, zimin, zimax, x_sub, y_sub, z_sub


def main():
    # ---------- input -----------
    nx, ny = 32, 33
    xe = np.linspace(-10, 10, nx+1)
    #xe = np.logspace(-1, 1, nx+1)
    ye = np.linspace(-10, 10, ny+1)
    xc = 0.5 * (xe[:-1] + xe[1:])
    yc = 0.5 * (ye[:-1] + ye[1:])
    xx, yy = np.meshgrid(xc, yc)
    # ----------------------------


    # ---------- debug ------------
    '''
    # 2D
    # model on an input grid
    dd = np.exp( - (xx**2. / 18.) - (yy**2. / 18.))

    # nested grid
    gridder = Nested2DGrid(xx,yy)
    xx_sub, yy_sub = gridder.nest([-3., 3.], [-3., 3.], 2)
    # model on the nested grid
    dd_sub = np.exp( - (xx_sub**2. / 18.) - (yy_sub**2. / 18.))
    # binned
    dd_binned = gridder.binning_onsubgrid(dd_sub)
    dd_re = dd.copy()
    #print(gridder.where_subgrid())
    dd_re[gridder.where_subgrid()] = dd_binned.ravel()



    # plot
    fig, axes = plt.subplots(1,3)
    ax1, ax2, ax3 = axes

    xx_plt, yy_plt = np.meshgrid(xe, ye)
    ax1.pcolor(xx_plt, yy_plt, dd, vmin = 0., vmax = 1.)
    #ax1.pcolor(xx_sub_plt, yy_sub_plt, dd_sub)

    xx_sub_plt, yy_sub_plt = np.meshgrid(gridder.xe_sub, gridder.ye_sub)
    ax2.pcolor(xx_sub_plt, yy_sub_plt, dd_sub, vmin = 0., vmax = 1)
    c = ax3.pcolor(xx_plt, yy_plt, dd - dd_re, vmin = 0., vmax = 1)


    for axi in [ax1, ax2, ax3]:
        axi.set_xlim(-10,10)
        axi.set_ylim(-10,10)

    for axi in [ax2, ax3]:
        axi.set_xticklabels('')
        axi.set_yticklabels('')

    cax = ax3.inset_axes([1.03, 0., 0.03, 1.])
    plt.colorbar(c, cax=cax)
    plt.show()
    '''


    # 1D
    # model on an input grid
    model = lambda x: np.exp( - (x**2. / 18.))
    d = model(xc)
    # nested grid
    nstg1D = Nested1DGrid(xc)
    x_sub = nstg1D.nest(3)
    # model on the nested grid
    d_sub = model(x_sub)
    # binned
    d_binned = nstg1D.binning_onsubgrid(d_sub)

    # plot
    fig = plt.figure()
    ax = fig.add_subplot(111)

    for xi, di, label, ls in zip(
        [xc, x_sub, xc],
        [d, d_sub, d_binned],
        ['Original', 'Subgrid', 'Binned'],
        ['-', '-', '--']
        ):
        ax.step(xi, di, where = 'mid', lw = 2., alpha = 0.5, ls = ls)

    plt.show()



if __name__ == '__main__':
    main()