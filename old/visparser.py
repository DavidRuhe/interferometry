import argparse
import struct
import numpy as np
import os
import ctypes

LEN_HDR = 512
CORR_HDR_MAGIC = 0x000000003B98F002


def instrument_config(instrument):
    return {"aartfaac6": (288), "aartfaac12": (576),}.get(instrument, None)

class VisParser:
    def __init__(self, instrument):
        self.instrument = instrument
        self.n_ant = instrument_config(instrument)

    @staticmethod
    def _parse_header(hdr):
        """ Parse the headers of raw visibilities as generated by the correlator.
            Set other fields to NaNs.
        (from aartfaac-calibration-pipeline/src/server/packet.h)
        struct input_header_t
        {
            uint32_t magic;       ///< magic to determine header
            uint32_t pad0;        ///< first padding
            double   startTime;   ///< start time (unix)
            double   endTime;     ///< end time (unix)
            uint32_t weights[78]; ///< station weights n*(n+1) / 2, where n in 
                                    {6, 12}
            uint8_t  pad1[176];   ///< 512 byte block
        };

        Record layout: Raw correlator visibilities for a single timeslice, and 
            thus containing all channels for both pols, have a single header.
            Frequency inforation is absent, so is array configuration and number
            of antennas. These have to be explicitly provided.
        """
        magic, t0, t1 = struct.unpack("<Qdd", hdr[0:24])
        magic = ctypes.c_uint32(magic).value
        nweight = (12 * 13) // 2
        weights = struct.unpack(("<%di" % nweight), hdr[148 : 148 + nweight * 4])

        # The other fields returned by the parser for calibrated data are set to
        # zeros in the class initialiser, and are not updated by this function.
        return (magic, t0, t1, weights)

    def __call__(self, file, subband):

        freq = subband * 195312.5  # Hz

        with open(file, "rb") as f:
            header = f.read(LEN_HDR)

            magic, t_start, t_end, weights = self._parse_header(header)

            # Check if magic number corresponds to the correlator.
            assert magic == CORR_HDR_MAGIC

            size = os.path.getsize(f.name)

            # Number of baselines
            n_bsl = self.n_ant * (self.n_ant + 1) / 2

            # Number of baselines times 8 bits.
            len_bdy = n_bsl * 8

            # Array Correlation Matrix
            n_acms = size // (LEN_HDR + len_bdy)
            assert n_acms.is_integer()
            print(len_bdy)
            print(f"Number of Array Correlation Matrices: {n_acms}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "files",
        metavar="FILE",
        type=str,
        nargs="+",
        help="Files containing calibrated visibilities, supports glob patterns",
    )
    parser.add_argument(
        "--instrument", type=str, choices=["aartfaac6", "aartfaac12"], required=True
    )
    parser.add_argument("--subband", type=int, required=True)
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

    parser = VisParser(args.instrument)
    for file in args.files:
        parser(file, args.subband)

