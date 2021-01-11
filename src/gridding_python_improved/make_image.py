import os
import numpy as np
import struct
import matplotlib.pyplot as plt
import gfft

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

    data = np.reshape(np.frombuffer(data, dtype=np.complex64), (NBLINE, NCHAN, POL2REC))
    return data


# xx, xy, yx, yy
with open(FNAME, 'rb') as f:
    f.seek(0 * RECSIZE)

    vis = read_vis(f)
data = vis.copy()
data.shape

from parse_lba_antennafield import parse_lba_antennafield
L = parse_lba_antennafield('../../antennafields/a12-AntennaField.conf', [SUB])

from scipy.constants import c
meters_to_wavelengths = 2 * SUB * 2e8 / (1024 * c)  # frequency over speed is wavelength

L_2d = L[:, :2]
distances = meters_to_wavelengths * (L_2d[:, None] - L_2d)

u, v = distances[np.tril_indices(576)].T

res = 2300
dx = 1.0 / res
out_ax = [(dx, res), (dx, res)]

L = np.linspace(-1, 1, res)
M = np.linspace(-1, 1, res)
mask = np.ones((res, res))
xv, yv = np.meshgrid(L, M)


image = data.copy()
im = image.mean(axis=(-1, -2))

im = np.rot90(np.real(gfft.gfft(np.ravel(im), [u, v], out_ax, enforce_hermitian_symmetry=True, verbose=True, W=6, alpha=1.5)), 1)*mask


plt.imsave('image.png', im, vmin=np.nanmean(im) - 3 * np.nanstd(im), vmax=np.nanmean(im) + 3 * np.nanstd(im), origin='lower')

plt.figure(figsize=(16, 16), facecolor='white')
plt.imshow(im, vmin=np.nanmean(im) - 3 * np.nanstd(im), vmax=np.nanmean(im) + 3 * np.nanstd(im), origin='lower')
plt.title('SB320-202012281240-lba_outer.vis direct imaging')
plt.savefig('image.png')