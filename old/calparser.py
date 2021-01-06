import argparse
from pysquasher import LEN_BDY
import struct
import numpy as np
import os

LEN_HDR = 512
A6_HDR_MAGIC = 0x4141525446414143

class CalParser:

    def __init__(self, instrument):
        self.instrument = instrument


    def parse_header(self, hdr):
        """
        Parse aartfaac header for calibrated data
        struct output_header_t
        {
        uint64_t magic;                   ///< magic to determine header                (  8 B)
        double start_time;                ///< start time (unix)                        (  8 B)
        double end_time;                  ///< end time (unix)                          (  8 B)
        int32_t subband;                  ///< lofar subband                            (  4 B)
        int32_t num_dipoles;              ///< number of dipoles (288 or 576)           (  4 B)
        int32_t polarization;             ///< XX=0, YY=1                               (  4 B)
        int32_t num_channels;             ///< number of channels (<= 64)               (  4 B)
        float ateam_flux[5];              ///< Ateam fluxes (CasA, CygA, Tau, Vir, Sun) ( 24 B)
        std::bitset<5> ateam;             ///< Ateam active                             (  8 B)
        std::bitset<64> flagged_channels; ///< bitset of flagged channels               (  8 B)
        std::bitset<576> flagged_dipoles; ///< bitset of flagged dipoles                ( 72 B)
        uint32_t weights[78];             ///< stationweights n*(n+1)/2, n in {6, 12}   (312 B)
        uint8_t pad[48];                  ///< 512 byte block                           ( 48 B)
        };
        """
        m, t0, t1, s, d, p, c = struct.unpack("<Qddiiii", hdr[0:40])
        # Flagged dipoles.
        f = np.frombuffer(hdr[80:152], dtype=np.uint64)
        return m, t0, t1, s, d, p, c, f


    def check_instrument(self, magic):
        if self.instrument == 'a6':
            assert magic == A6_HDR_MAGIC

        elif self.instrument == 'a12':
            raise NotImplementedError()

        else:
            raise ValueError()


    def __call__(self, file):
        with open(file, 'rb') as f:
            header = f.read(LEN_HDR)
            magic, t_start, _, subband, n_ant, _, _, _ = self.parse_header(header)
            # Check if magic number corresponds to the telescope.
            self.check_instrument(magic)

            size = os.path.getsize(f.name)

            # Number of baselines
            n_bsl = n_ant * (n_ant + 1) / 2

            # Number of baselines times 8 bits.
            len_bdy = n_bsl * 8

            # Array Correlation Matrix
            n_acms = size // (LEN_HDR + len_bdy)
            assert n_acms.is_integer()
            print(f"Number of Array Correlation Matrices: {n_acms}")








if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('files', metavar='FILE', type=str, nargs='+',
            help="Files containing calibrated visibilities, supports glob patterns")
    parser.add_argument('--instrument', type=str, choices=['a6', 'a12'], required=True)
    # parser.add_argument('--nant', type=int, help="Number of antennas", required=True)
    # parser.add_argument('--res', type=int, default=1024,
            # help="Image output resolution (default: %(default)s)")
    # parser.add_argument('--window', type=int, default=6,
    #         help="Kaiser window size (default: %(default)s)")
    # parser.add_argument('--alpha', type=float, default=1.5,
    #         help="Kaiser window alpha param (default: %(default)s)")
    # parser.add_argument('--inttime', type=int, default=1,
    #         help="Integration time (default: %(default)s)")
    # parser.add_argument('--antpos', type=str, default='/usr/local/share/aartfaac/antennasets/lba_outer.dat',
    #         help="Location of the antenna positions file (default: %(default)s)")
    # parser.add_argument('--nthreads', type=int, default=multiprocessing.cpu_count(),
    #         help="Number of threads to use for imaging (default: %(default)s)")
    # parser.add_argument('--output', type=str, default=os.getcwd(),
    #         help="Output directory (default: %(default)s)")

    args = parser.parse_args()

    parser = CalParser(args.instrument)
    for file in args.files:
        parser(file)

