from scipy.constants import c
import numpy as np

def parse_lba_antennafield(f, subbands):
    """"""

    with open(f) as f:
        lines = [line.strip() for line in f.readlines()]

    positions = []
    lba_start = lines.index('LBA')
    data_start = lba_start + 3
    data_end = lines.index(']', lba_start)
    num_antennas = data_end - data_start
    offset = [float(x) for x in lines[lba_start+1].split()[2:5]]
    for line in lines[data_start: data_end]:
        x, y, z = [float(x) for x in line.split()[0:3]]
        positions.append(
            [offset[0] + x, offset[1] + y, offset[2] + z]
        )

    rot_matrix = []
    lba_rot_start = lines.index('ROTATION_MATRIX LBA')
    data_start = lba_rot_start + 2
    data_end = lba_rot_start + 5
    for line in lines[data_start: data_end]:
        x, y, z = [float(x) for x in line.split()[0:3]]
        rot_matrix.append(
            [x, y, z]
        )

    A = np.array(positions)
    R = np.array(rot_matrix)
    L = A.dot(R)
    
    return L

    u, v = [], []

    for a1 in range(0, num_antennas):
        for a2 in range(0, a1+1):
            u.append(L[a1, 0] - L[a2, 0])
            v.append(L[a1, 1] - L[a2, 1])

    # Converts subband to frequency. 
    return L, [np.ravel([(np.array(u)/(c/(s*(2e8/1024))/2.0)) for s in subbands]),
            np.ravel([(np.array(v)/(c/(s*(2e8/1024))/2.0)) for s in subbands])]