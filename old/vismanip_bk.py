''' Collection of classes for manipulation of visibilities generated by
    AARTFAAC
    pep/11Oct16
'''
import sys
import os
import struct
import numpy as np
import datetime
import pytz
try:
    import casacore.tables as pytab
    import casacore.measures as pymeas
except ImportError:
    print ('vismanip: python casacore not found! Bailing out...')
    # sys.exit(-1);

CORR_HDR_MAGIC= 0x000000003B98F002 # Magic number in raw corr visibilities.
A06_HDR_MAGIC = 0x4141525446414143
A12_HDR_MAGIC = 0x4141525446414144
LEN_HDR       = 512

class TransitVis (object):
    fname       = None # Name of visibility binary file.
    hdr         = None # record header
    fvis        = None # File pointer to binary file
    nrec        = None # Total number of records in this file
    nbline      = None # Total number of baselines in this file
    tfilestart  = None # First timestamp in the file
    tfileend    = None # Last timestamp in the file

    # Information extracted from an individual header
    magic       = None
    trec        = None # Current records start time
    sub         = None # Subband number
    nant        = None # Total number of dual-pol antennas in the array.
    npol        = None # Number of polarizations in the observation
    pol2rec     = None # Number of polarizations per record.
    nchan       = None # Number of channels in this file
    norigchan   = None # Number of original channels before preprocessing.
    dt          = None # Integration time per record

    # Per polarization information
    flagged_channels= None # Independent per pol.
    flagged_dipoles = None # Independent per pol.
    ateamflux   = None # Independent per pol.
    weights     = None
    nweight     = None

    # Frequency associated with channel or averaged set of channels.
    freq        = None 

    recbuf      = None # Complete, single record read from file
    vis         = None # Usable visibility array per record (both pols)
    tstamps     = None # List containing timestamps of all records in a file.

    deblev      = None # Debug level for messages.
    missrec     = None # Number of records thrown due to mismatched times.

    # Function abstraction for raw and calibrated visibilities
    read_rec    = None # Function object, initialised to the right read func.
    read_hdr    = None # Function object, initialised to the right read func.
    parse_hdr   = None # Function object, initialised to the right parse func.

    def __init__ (self, fname, nant=None, sub=None, nchan=None, 
                    arrayconfig=None):
        """ Initialization of internal state corresponding to a visibility dump.
            Since the correlator raw visibility magic number doesn't distinguish
            between AARTFAAC-6 or AARTFAAC-12, and has no subband information, 
            this can be supplied to the init routine.
        """
        # Check if fname is a valid file, and open
        self.fname = fname
        self.fvis = open (self.fname, 'rb')
        self.hdr = self.fvis.read (LEN_HDR)
        self.npol = 2
        print ('<-- Warning: Number of polarizations hardcoded to 2 for now.')
        return

        self.magic,self.tfilestart,trec = struct.unpack ("<Qdd", self.hdr[0:24])

        print(self.tfilestart)

        if (self.magic == A06_HDR_MAGIC):
            self.nant     = 288
            self.read_rec = self.read_cal
            self.parse_hdr= self.parse_cal_hdr
            self.read_hdr = self.read_cal_hdr
            self.nchan    = 1 # See record description in parse_cal_hdr
            self.nweight  = (6*7)/2 # Per station weights
            self.pol2rec  = npol

        elif (self.magic == A12_HDR_MAGIC):
            self.nant     = 576
            self.read_rec = self.read_cal
            self.parse_hdr= self.parse_cal_hdr
            self.read_hdr = self.read_cal_hdr
            self.nchan    = 1 # See record description in parse_cal_hdr
            self.nweight  = (12*13)/2 # Per station weights
            self.pol2rec  = npol

        elif (self.magic == CORR_HDR_MAGIC):
            assert nant != None, \
                '### Please specify the number of antennas in the .vis file.'
            assert sub != None,  \
                '### Please specify subband information for this .vis file.'
            assert arrayconfig != None,  \
                '### Please specify the array configuration information for \
                this .vis file.'
            assert nchan != None,  \
                '### Please specify the number of channels for this .vis file.'
            self.nant = nant
            if self.nant == 288:
                self.nweight =(6*7)/2
            elif self.nant == 576:
                self.nweight =(12*13)/2
            self.sub       = sub
            self.parse_hdr = self.parse_vis_hdr
            self.read_rec  = self.read_vis
            self.read_hdr = self.read_vis_hdr
            self.nchan     = nchan
            self.norigchan = nchan
            self.pol2rec  = 2

        self.nchan = 1
        self.nant = 576

        self.parse_hdr = self.parse_vis_hdr
        self.nweight =(12*13)/2

        # Old format, storing full float ACM
        # self.payload = self.nant**2 * 8 

        # New format, stores only upper triangle float ACM
        print('hi')
        self.nbline  = self.nant*(self.nant+1)/2
        self.pol2rec = 2
        # 8 due to float complex.
        self.payload = self.nbline * self.nchan * self.pol2rec * 8
        self.recsize = self.payload + LEN_HDR
        fsize = os.path.getsize(fname)

        # NOTE: We need to read 2 recs for getting both polarizations from .cal
        if self.magic == A12_HDR_MAGIC or self.magic == A06_HDR_MAGIC:
            self.nrec = fsize/(self.recsize*2)
        else:
            self.nrec = fsize/(self.recsize) # each record has both pols in .vis
            # First record has a corrupted timestamp in a .vis file
            print(self.recsize)
            # self.fvis.seek (int(self.recsize), 0) 
            # self.hdr = self.fvis.read (LEN_HDR)
            # self.magic,self.tfilestart,trec = struct.unpack ("<Qdd", 
            #                                             self.hdr[0:24]

        self.fvis.seek (0)
        self.dt = datetime.timedelta (seconds=(trec - self.tfilestart))# seconds

        self.tfilestart = datetime.datetime.utcfromtimestamp(self.tfilestart)\
                        .replace(tzinfo= pytz.utc)
        
        # Find the last timestamp in a file.
        self.fvis.seek (0)
        self.fvis.seek (int((self.nrec-1) * self.recsize))
        self.hdr = self.fvis.read (LEN_HDR)
        self.magic, self.tfileend, trec, self.sub, self.nant, pol, \
            self.norigchan, _, _, _, _, self.freq = self.parse_hdr (self.hdr)

        self.fvis.seek (0)

        # Initialize data structures
        # Hardcoded to 5 sources
        self.ateamflux       = np.zeros ( (self.npol, 5) ) 
        # 64-sized bitset.
        self.flagged_channels= np.zeros ( (self.npol, 8), dtype=np.int8) 
        # 576-sized bitset.
        self.flagged_dipoles = np.zeros ( (self.npol, 72), dtype=np.int8) 
        # Weights are not independent per pol.Size depends on number of ants
        self.weights         = np.zeros ( (1, self.nbline), dtype=np.int32) 
        self.vis             = np.zeros ( (self.nbline, self.nchan, self.npol),
                                         dtype=np.complex64)
        self.missrec         = 0
        self.tstamps         = []

    def __del__ (self):
        if (self.fvis):
            self.fvis.close ()

    def __str__ (self):
        return '# Time: \n#  File start: %s\n#  File end  : %s\n#  Rec: %s\n\
# Subband: %03d, pols : %d, chans: %d\n# Num. recs: %04d\n'\
         % (self.tfilestart, self.tfileend,  self.trec, self.sub, self.npol, 
            self.nchan, self.nrec)

    def parse_vis_hdr (self, hdr):
        ''' Parse the headers of raw visibilities as generated by the correlator.
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
        '''
        try:
            magic, t0, t1 = struct.unpack ("<Qdd", hdr[0:24])
            print(t0, t1, magic)
            print('joe')
            weights = struct.unpack (("<%di" % self.nweight), 
                                    hdr[148:148+self.nweight*4])
        except struct.error:
            print ('### struct unpack error!')
            raise

        assert magic == CORR_HDR_MAGIC
        freq = self.sub * 195312.5 # Hz
        # The other fields returned by the parser for calibrated data are set to
        # zeros in the class initialiser, and are not updated by this function.
        return (magic, datetime.datetime.utcfromtimestamp(t0).replace(tzinfo=\
        pytz.utc), datetime.datetime.utcfromtimestamp(t1).replace(tzinfo=\
        pytz.utc), self.sub, self.nant, self.npol, self.norigchan, 
        self.ateamflux, 0, 0, weights, freq)

    def parse_cal_hdr(self, hdr):
        """
        Parse aartfaac header for calibrated data, independent per pol.
        Two pols are packed one after another, but with independent headers.

        struct output_header_t
        {
          uint64_t magic;                   ///< magic to determine header
          double start_time;                ///< start time (unix)
          double end_time;                  ///< end time (unix)
          int32_t subband;                  ///< lofar subband
          int32_t num_dipoles;              ///< number of dipoles (288 or 576)
          int32_t polarization;             ///< XX=0, YY=1
          int32_t num_channels;             ///< number of channels (<= 64)
          float ateam_flux[5];              ///< Ateam fluxes(CasA, CygA, Tau, Vir, Sun)
          std::bitset<5> ateam;             ///< Ateam active
          std::bitset<64> flagged_channels; ///< bitset of flagged channels (8 byte)
          std::bitset<576> flagged_dipoles; ///< bitset of flagged dipoles (72 byte)
          uint32_t weights[78];             ///< stationweights n*(n+1)/2, n in {6, 12}
          uint8_t pad[48];                  ///< 512 byte block
        };

        Record layout: The incoming channels in the raw visibilities from the
            correlator can be averaged in various combinations. The resulting
            averaged product gets its own record with an independent header.
            The record size for all records is the same, and corresponding data
            products (all records with the same channel integration, e.g.) will
            go into a single file. Other channel groups being averaged will go 
            into separate files.  
        """
        try:
            magic, t0, t1, sub, nant, pol, norigchan = struct.unpack("<Qddiiii",
                                                     hdr[0:40])
            ateamflux = struct.unpack("<fffff", hdr[40:60])
            ateam, flag_chans = struct.unpack("<QQ", hdr[60:76])
            flag_dips = struct.unpack ("<72b", hdr[76:148])
            weights = struct.unpack (("<%di" % self.nweight), 
                                    hdr[148:148+self.nweight*4]) # int weights
        except struct.error:
            print ('### struct unpack error!')
            raise

        assert (magic == A06_HDR_MAGIC or magic == A12_HDR_MAGIC)
        # TODO: Examine the flagged channels to determine the actual frequency 
        # of the averaged channels. Currently return subband center.
        freq = sub * 195312.5 # Hz, 
        return (magic, datetime.datetime.utcfromtimestamp(t0).replace(tzinfo=
        pytz.utc), datetime.datetime.utcfromtimestamp(t1).replace(tzinfo=
        pytz.utc), sub, nant, pol, norigchan, ateamflux, flag_chans, flag_dips, 
        weights, freq)

    # Return the header for a given record number.
    # If record number is specified as None, the next record will be returned.
    def rec_hdr (self, rec):
        return hdr

    def read_vis_hdr (self, rec):
        
        assert (self.fvis)
        if (rec != None):
            raise NotImplemented

        hdr = self.fvis.read (LEN_HDR)
        m, self.trec, t1, s, d, p, c, ateamflux, flag_chan, flag_dip, weights, \
                 freq = self.parse_vis_hdr (hdr)
        self.fvis.read (self.payload)
        return 0

    # Return just the header, with metainformation extracted.
    # Faster than doing an actual read of the record payload.
    def read_cal_hdr (self, rec):
        
        assert (self.fvis)
        if (rec != None):
            raise NotImplemented

        # First pol.
        hdr = self.fvis.read (LEN_HDR)
        m, self.trec, t1, s, d, p, c, ateamflux, flag_chan, flag_dip, weights, \
                 freq = self.parse_hdr (hdr)
        self.ateamflux[p] = ateamflux
        self.flagged_dipoles[p] = flag_dip
        self.flagged_channels[p] = flag_chan

        self.fvis.seek (self.payload, 1)

        # Second polarization
        hdr = self.fvis.read (LEN_HDR)
        self.magic, trec, _, sub, self.nant, pol, \
            self.norigchan, ateamflux, flag_chan, flag_dip, weights, \
                    freq = self.parse_hdr (hdr)

        self.fvis.seek (self.payload, 1)
        # Check if both records correspond to the same trec
        # We can choose to keep only the single pol, or reject this record.
        if  (trec != self.trec):
            self.recbuf = 0
            self.fvis.seek (-self.recsize, 1) 
            print ('## Mismatched times: pol %d tim: %s, pol %d tim: %s' \
                    % (p,self.trec, pol, trec))
            self.missrec = self.missrec + 1
            return -1

        self.ateamflux[p] = ateamflux
        self.flagged_dipoles[p] = flag_dip
        self.flagged_channels[p] = flag_chan

        return 1

    def read_vis (self, rec):
        """ Routine to read a single timestamp from a raw visibility file 
            generated by the correlator.
        """
        assert (self.fvis)
        if (rec != None):
            raise NotImplemented

        self.recbuf = self.fvis.read (self.recsize)
        m, self.trec, t1, _,_,_,_,_,_,_,_,_ = self.parse_vis_hdr (\
                                                    self.recbuf[0:LEN_HDR])
        self.vis =  np.reshape (np.fromstring(self.recbuf[LEN_HDR:], 
                     dtype=np.complex64), (self.nbline, self.nchan, self.npol) )

        return self.vis

    # Routine to return the record for a given record number.
    # If record number is specified as None, the next record will be returned.
    def read_cal (self, rec):

        assert (self.fvis)
        if (rec != None):
            raise NotImplemented

        # First polarization
        self.recbuf = self.fvis.read (self.recsize)
        m, self.trec, t1, s, d, p, c, ateamflux, flag_chan, flag_dip, weights, \
                  freq  = self.parse_hdr (self.recbuf[0:LEN_HDR])
        self.ateamflux[p] = ateamflux
        self.flagged_dipoles[p] = flag_dip
        self.flagged_channels[p] = flag_chan

        # Hardcoded 0 due to the presence of a single channel in calibrated data
        self.vis[:,0,p]=np.fromstring(self.recbuf[LEN_HDR:], dtype=np.complex64)

        # Second polarization
        self.recbuf = self.fvis.read (self.recsize)
        self.magic, trec, _, sub, self.nant, pol, \
            self.norigchan, ateamflux, flag_chan, flag_dip, weights, \
                freq  = self.parse_hdr (self.recbuf[0:LEN_HDR])

        # Check if both records correspond to the same trec
        # We can choose to keep only the single pol, or reject this record.
        if  (trec != self.trec):
            self.recbuf = 0
            self.fvis.seek (-self.recsize, 1) 
            print ('## Mismatched times: pol %d tim: %s, pol %d tim: %s' \
                    % (p,self.trec, pol, trec))
            self.missrec = self.missrec + 1
            return None

        self.ateamflux[p] = ateamflux
        self.flagged_dipoles[p] = flag_dip
        self.flagged_channels[p] = flag_chan
        self.vis[:,0,pol] =  np.fromstring(self.recbuf[LEN_HDR:], dtype=np.complex64)

        return self.vis

    def extract_timestamps (self, recstart=None, recend=None):
        """ Extract timestamps of records in a given range in the associated file
            and return as a datetime list. If the range is not specified (None), 
            extract for the whole file. 
        """
        if recstart is None:
            recstart = 0
        if recend is None:
            recend = self.nrec

        assert (recstart < recend)
        assert (self.fvis)
        self.fvis.seek(0)
        self.fvis.seek(self.recsize*recstart)

        ind = 0
        tick = int ((recend-recstart)/10)
        while ind < (recend-recstart):
            try:
                dat = self.read_hdr (None)
                if dat == -1:
                    print ('<-- Times of both pol records didnt match! Ignoring \
                            this record...')
                    pass
            except:
                print ('### Error in trying to read record from file ', self.fname)
                break
        
            self.tstamps.append (self.trec)
            ind = ind + 1;
            if ind%tick == tick-1:
                print ('\r<-- %05d of %d recs done.' % (ind, self.nrec))

        # There can be differences at the 10^-4 sec level in the time differences.
        tdel = [x.seconds for x in np.diff (self.tstamps)] 
        tmissidx = np.where (tdel != self.dt.seconds)

        print ('<-- Read %04d records of %04d.\n' % (ind, self.nrec))
        print ('<-- Discarded %d recs in specified range due to time mismatches \
                between pols.' % self.missrec)
        print ('<-- Found %d time jumps > integration time in the specified \
                range.' % len(tmissidx))
        
        return self.tstamps

    # Return the record corresponding to a given timestamp
    def read_time (self, tim):
        # Find the record number corresponding to this time
        trec = None
        return self.read_rec (trec)

    def create_ms_ant_table (self):
        return None

    def create_ms (self):
        return None