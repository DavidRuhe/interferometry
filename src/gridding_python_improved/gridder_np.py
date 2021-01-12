# import torch
import math
# from gridding import gcf_kaiser, inv_gcf_kaiser
# from gridding_python_improved.add_at import add_at
import numpy as np
# from gridding_python_improved._fft import fftshift, ifftshift
# from torch.fft import fftn

# torch.set_printoptions(sci_mode=False)

A = [
    -4.41534164647933937950E-18,
    3.33079451882223809783E-17,
    -2.43127984654795469359E-16,
    1.71539128555513303061E-15,
    -1.16853328779934516808E-14,
    7.67618549860493561688E-14,
    -4.85644678311192946090E-13,
    2.95505266312963983461E-12,
    -1.72682629144155570723E-11,
    9.67580903537323691224E-11,
    -5.18979560163526290666E-10,
    2.65982372468238665035E-9,
    -1.30002500998624804212E-8,
    6.04699502254191894932E-8,
    -2.67079385394061173391E-7,
    1.11738753912010371815E-6,
    -4.41673835845875056359E-6,
    1.64484480707288970893E-5,
    -5.75419501008210370398E-5,
    1.88502885095841655729E-4,
    -5.76375574538582365885E-4,
    1.63947561694133579842E-3,
    -4.32430999505057594430E-3,
    1.05464603945949983183E-2,
    -2.37374148058994688156E-2,
    4.93052842396707084878E-2,
    -9.49010970480476444210E-2,
    1.71620901522208775349E-1,
    -3.04682672343198398683E-1,
    6.76795274409476084995E-1
]


B = [
    -7.23318048787475395456E-18,
    -4.83050448594418207126E-18,
    4.46562142029675999901E-17,
    3.46122286769746109310E-17,
    -2.82762398051658348494E-16,
    -3.42548561967721913462E-16,
    1.77256013305652638360E-15,
    3.81168066935262242075E-15,
    -9.55484669882830764870E-15,
    -4.15056934728722208663E-14,
    1.54008621752140982691E-14,
    3.85277838274214270114E-13,
    7.18012445138366623367E-13,
    -1.79417853150680611778E-12,
    -1.32158118404477131188E-11,
    -3.14991652796324136454E-11,
    1.18891471078464383424E-11,
    4.94060238822496958910E-10,
    3.39623202570838634515E-9,
    2.26666899049817806459E-8,
    2.04891858946906374183E-7,
    2.89137052083475648297E-6,
    6.88975834691682398426E-5,
    3.36911647825569408990E-3,
    8.04490411014108831608E-1
]


def chbevl(x, array, n):

    b0 = array[0]
    b1 = 0
    i = 1
    j = n - 1

    while True:
        b2 = b1
        b1 = b0
        b0 = x * b1 - b2 + array[i]
        i += 1
        j -= 1

        if j == 0:
            break

    return 0.5 * (b0 - b2)


def i0e(x):

    x[x < 0] *= 1

    y = np.zeros_like(x)

    se8 = x <= 8.0

    y[se8] = chbevl(x[se8] / 2 - 2, A, 30)
    xg8 = x[~se8]
    y[~se8] = chbevl(32 / xg8 - 2, B, 25) / np.sqrt(xg8)
    return y


def i0(x):
    return np.exp(x) * i0e(x)


def _gcf_kaiser(k, dk, beta):
    temp3 = 2 * k / dk

    # if torch.any((1 - temp3)*(1 + temp3) < -1e-12):
    #     raise Exception("There is an issue with the gridding code!")

    temp3 = np.sqrt(abs((1 - temp3) * (1 + temp3)))

    temp3 = beta * temp3

    C = i0(temp3) / i0(beta)
    return C


def _inv_gcf_kaiser(x, dk, W, beta):

    temp1 = math.pi * math.pi * W * W * dk * dk * x * x
    temp2 = beta * beta

    temp = np.sqrt(temp2 - temp1)
    c0 = (np.exp(beta)-np.exp(-1.*beta))/2./beta
    c = (np.exp(temp) - np.exp(-1.*temp))/2./temp

    tdg = temp1 > temp2

    if tdg.any():
        temp = temp1[tdg] - temp2
        c[tdg] = -0.5 * (np.exp(-1 * temp) - np.exp(temp)) / temp

    return c / c0


class Gridder:
    def __init__(self, config):
        self.l = np.linspace(-1, 1, config['resolution'])
        self.m = np.linspace(-1, 1, config['resolution'])
        self.xv, self.yv = np.meshgrid(self.l, self.m)

        self.config = config
        self.nx = config['resolution']
        self.dx = 1 / config['resolution']

        self.ny = config['resolution']
        self.dy = 1 / config['resolution']

        self.du = 1 / config['alpha']
        self.nu = int(self.nx / self.du)

        self.dv = 1 / config['alpha']
        self.nv = int(self.ny / self.dv)

        self.vmin = -0.5 * config['resolution']
        self.umin = -0.5 * config['resolution']

        self.xmin = -0.5
        self.ymin = -0.5

        self.beta = np.array([self.get_beta()])

    def get_beta(self):
        # see Beatty et al. (2005)
        W = self.config['W']
        alpha = self.config['alpha']
        beta = math.pi * math.sqrt((W * W / alpha / alpha) *
                                   (alpha - 0.5) * (alpha - 0.5) - 0.8)
        return beta

    def grid_1d_from_2d(self, x, dx, vis, y):
        print(x.sum())
        nvis, N = x.shape
        W = self.config['W']
        Dx = W * dx

        xref = np.ceil((x - 0.5 * W * dx) / dx) * dx
        np.save('xref.npy', xref)
        xndx = np.arange(W)
        print(xndx)
        print(x.sum())
        np.save('x.npy', x)
        print(dx)
        np.save('xndx.npy', xndx)
        xg = xref + xndx * dx
        print(xg.sum())
        np.save('dx.npy', dx)
        np.save('xg.npy', xg)
        np.save('xgminx.npy', xg - x)
        gcf_val = _gcf_kaiser(xg - x, Dx, self.beta)

        np.save('gcf_val.npy', gcf_val)

        vis2 = np.matmul(vis[:, :, None], gcf_val[:, None, :].astype(np.complex64))
        vis2 = vis2.reshape(nvis, -1)
        x2 = np.tile(xg, N)
        y2 = np.repeat(y, W, axis=-1)
        return x2, vis2, y2

    def grid_2d(self, vis, u, v):

        np.save('unone.npy', u[:, None])

        tu1, tvis1, tv1 = self.grid_1d_from_2d(
            u[:, None], self.du, vis[:, None], v[:, None])

        np.save('tu1.npy', tu1)
        np.save('tvis1.npy', tvis1)
        np.save('tv1.npy', tv1)
        tv2, tvis2, tu2 = self.grid_1d_from_2d(tv1, self.dv, tvis1, tu1)  # output arrays

        np.save('tu2.npy', tu2)
        np.save('tvis2.npy', tvis2)
        np.save('tv2.npy', tv2)


        ug = tu2.reshape(-1)
        visg = tvis2.reshape(-1)
        vg = tv2.reshape(-1)

        np.save('visg.npy', visg)
        np.save('visgr.npy', visg.real)
        np.save('visgi.npy', visg.imag)

        gvi = np.zeros((self.nu, self.nv), dtype=np.complex64)

        undxi = ((ug - self.umin) / self.du + 0.5).astype(int)
        vndxi = ((vg - self.vmin) / self.du + 0.5).astype(int)

        mask = ((undxi >= 0) & (undxi < self.nu)) & ((vndxi >= 0) & (vndxi < self.nv))

        np.save('undxim.npy', undxi[mask])
        np.save('vndxim.npy', vndxi[mask])
        np.save('visgm.npy', visg[mask])

        np.add.at(gvi, (undxi[mask], vndxi[mask]), visg[mask])# print(gv.sum())

        print(gvi.sum(), 'gvi')
        np.save('gvi1.npy', gvi)

        hflag_u = self.config['hermitian_symmetry_u']
        hflag_v = self.config['hermitian_symmetry_v']

        if hflag_u or hflag_v:
            if hflag_u:
                undxi = ((-1.*ug - self.umin) / self.du + 0.5).astype(int)
            if hflag_v:
                vndxi = ((-1.*vg - self.vmin)/ self.dv + 0.5).astype(int)



            mask = ((undxi >= 0) & (undxi < self.nu)) & ((vndxi >= 0) & (vndxi < self.nv))    

            np.add.at(gvi, (undxi[mask], vndxi[mask]), visg.conjugate()[mask])

        np.save('gvi2.npy', gvi)

        return gvi


    def get_grid_corr_2d(self):
        x = np.arange(self.nx) * self.dx + self.xmin
        y = np.arange(self.ny) * self.dy + self.xmin

        W = self.config['W']
        x_inv_kaiser = _inv_gcf_kaiser(x, self.du, W, self.beta)
        y_inv_kaiser = _inv_gcf_kaiser(y, self.dv, W, self.beta)

        gridcorr = np.outer(x_inv_kaiser, y_inv_kaiser)
        return gridcorr


    def __call__(self, vis, u, v):
        input_grid = self.grid_2d(vis, u, v)
        np.save('input_grid.npy', input_grid)
        input_grid = np.fft.fftshift(input_grid)
        out = np.fft.fftn(input_grid)
        out = np.fft.fftshift(out)

        alpha = self.config['alpha']
        xl = int(0.5 * self.nx * (alpha - 1))
        yl = int(0.5 * self.nx * (alpha - 1))

        out = out[xl: xl + self.nx, yl: yl + self.ny]
        gc = self.get_grid_corr_2d()

        return out / gc


if __name__ == "__main__":

    import yaml

    with open('gridding_python_improved/settings.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print(config)

    import os
    import numpy as np
    import struct
    import matplotlib.pyplot as plt

    # FNAME = '/project/druhe/SB320-202012281240-lba_outer.vis'
    FNAME = '/home/david/projects/interferometry/data/SB320-202012281240-lba_outer.vis'

    MAGIC = '0x000000003B98F002'
    int(MAGIC, 0)

    NANT = 576
    SUB = 320
    NCHAN = 1
    POL2REC = 4
    LEN_HDR = 512
    NBLINE = NANT * (NANT + 1) // 2
    PAYLOAD = NBLINE * NCHAN * POL2REC * 8
    RECSIZE = PAYLOAD + LEN_HDR

    fsize = os.path.getsize(FNAME)
    nrec = fsize // RECSIZE
    nrec

    def read_vis(f):
        s = f.read(RECSIZE)

    #     header = s[:512]
        data = s[512:]

        magic, = struct.unpack("<Q", s[0:8])
        magic = np.uint32(magic)
        assert int(magic) == int(MAGIC, 0)

        data = np.reshape(np.frombuffer(data, dtype=np.complex64),
                          (NBLINE, NCHAN, POL2REC))
        return data

    # xx, xy, yx, yy
    with open(FNAME, 'rb') as f:
        f.seek(0 * RECSIZE)

        vis = read_vis(f)
    data = vis.copy()
    data.shape

    from parse_lba_antennafield import parse_lba_antennafield
    L = parse_lba_antennafield('../antennafields/a12-AntennaField.conf', [SUB])

    from scipy.constants import c
    meters_to_wavelengths = 2 * SUB * 2e8 / \
        (1024 * c)  # frequency over speed is wavelength

    L_2d = L[:, :2]
    distances = meters_to_wavelengths * (L_2d[:, None] - L_2d)


    u, v = distances[np.tril_indices(576)].T
    np.save('u.npy', u)

    gridder = Gridder(config)

    # u = torch.from_numpy(u)
    # v = torch.from_numpy(v)
    # vis = (vis - vis.mean()) / vis.std()
    # print(vis.sum())
    # vis = torch.from_numpy(vis)
    # print(vis.sum())

    vis = vis.mean((-1, -2))
    np.save('vis.npy', vis)
    np.save('v.npy', v)


    im = gridder(vis, u, v)
    im = np.rot90(im.real)
    # plt.imsave('image_np.png', im, vmin=np.mean(im) - 3 * np.std(im), vmax=np.mean(im) + 3 * np.std(im), origin='lower')
