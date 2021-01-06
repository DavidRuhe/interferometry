#!/usr/bin/env python

# Generate position files for different antennasets.
#
# The imaging pipeline needs to know the position of the AARTFAAC antennae.
#
# The LOFAR repository contains a per-station AntennaFieldCSXXX.conf file in
# the directory MAC/Deployment/data/StaticMetaData/AntennaFields. These
# provide information about the position of all LOFAR antennae. In particular,
# they contain a block that looks like:
#
#   LBA
#   3 [ XXXXX.XXXXX YYYYY.YYYYY ZZZZZ.ZZZZZ]
#   96 x 2 x 3 [
#     X.XXXX   Y.YYYY  Z.ZZZZ    X.XXXX  Y.YYYY Z.ZZZZ
#     X.XXXX   Y.YYYY  Z.ZZZZ    X.XXXX  Y.YYYY Z.ZZZZ
#     ...
#     X.XXXX   Y.YYYY  Z.ZZZZ    X.XXXX  Y.YYYY Z.ZZZZ
#   ]
#
# This tells us about all the LBA antennae in the station. The first three
# numbers provide the reference position of the station in IRTF2005. The
# subsequent array of 96 * 2 * 3 numbers provide per-antenna offsets from that
# reference. Each offset is repeated twice, for two polarizations, but the
# positions should be identical.
#
# Note that there are 96 antennae listed. The first 48 correspond to the
# LBA_INNER antennaset; the second 48 to LBA_OUTER. This is defined in
# MAC/Deployment/data/StaticMetaData/AntennaSets.conf; we take it as read for
# now.
#
# When the AARTFAAC correlator produces correlation matrices, it will order
# them such that we start with the first antenna being used in the CS002 file,
# and end with the last antenna in the CS007 file.
#
# The imaging pipeline requires a text file that lists a single IRTF2005
# X/Y/Z position per line. They should be ordered in the same way as the
# correlator output. That is, the first line contains the ITRF position of the
# first CS002 antenna in use, and the last line contains the position of the
# last CS007 antenna in use.
#
# This script processes the AntennaFieldCSXXX.conf files to generate output
# appropriate for AARTFAAC. Specify the type of antenna (LBA, HBA) and the
# range in use (0-48 for LBA_INNER, 48-96 for LBA_OUTER, 0-96 for
# LBA_SPARSE_EVEN, 1-96 for LBA_SPARSE_ODD) on the command line, together with
# one or more AntennaField files. E.g.:
#
#   $ python antennaset.py LBA 0 48 AntennaFieldCS002.conf AntennaFieldCS003.conf
#   $ python antennaset.py LBA 1 96 AntennaFieldCS002.conf AntennaFieldCS003.conf

import sys

class AntennaSet(object):
    def __init__(self, name, start_ant, end_ant, datafile):
        self.positions = []
        lines = [line.strip() for line in datafile.readlines()]
        lba_start = lines.index(name)
        data_start = lba_start + 3
        offset = [float(x) for x in lines[lba_start+1].split()[2:5]]
        step = 1
        if end_ant - start_ant > 48:
            step = 2
        for line in lines[data_start + start_ant:data_start + end_ant:step]:
            x, y, z = [float(x) for x in line.split()[0:3]]
            self.positions.append(
                [offset[0] + x, offset[1] + y, offset[2] + z]
            )

if __name__ == "__main__":
    name = sys.argv[1]                                   # LBA or HBA
    start_ant, end_ant = [int(x) for x in sys.argv[2:4]] # LBA_OUTER = 48,96
    antennasets = []
    # Remaining arguments are AntennaField files.
    for filename in sys.argv[4:]:
        with open(filename, "r") as f:
            antennasets.append(AntennaSet(name, start_ant, end_ant, f))
    for antset in antennasets:
        for posn in antset.positions:
            print("%f %f %f" % (posn[0], posn[1], posn[2]))