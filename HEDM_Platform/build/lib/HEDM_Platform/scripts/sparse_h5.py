import sys
import numpy as np
import time
import os
import functools
import numpy as np
import hdf5plugin
import h5py
import numba
import fabio
import concurrent.futures
from ImageD11 import sparseframe, cImageD11

bg_scale = 0
file_path = os.path.join(sys.argv[5], 'removalmask.npy')
removalmask = np.load(file_path)

class Options:
    HOWMANY = int(2048*2048*0.25)
    THRESHOLDS = (int(sys.argv[3]), 97, 98, 99, 100)
    PIXELS_IN_SPOT = 3
    SCANMOTORS = ("diffrz", "diffrz_center")
    HEADERMOTORS = ("diffty", "difftz" "samtx", "samty", "samtz", "bigy")
    DETECTORNAME = "frelon3"
    MASKFILE = None
    BACKGROUND = sys.argv[2]
    CORES = os.cpu_count() - 1
    HNAME = sys.argv[1]
    OUTPATH = os.getcwd()

    def __init__(self):
        if self.MASKFILE:
            self.MASK = 1 - fabio.open(self.MASKFILE).data
        else:
            self.MASK = None
        self.BG = scaledbg(fabio.open(self.BACKGROUND).data) if self.BACKGROUND else None

    def correct(self, frm):
        cor = frm
        if self.MASK is not None:
            cor *= self.MASK
        if self.BG is not None:
            cor = cor.astype(np.float32) - self.BG.estimate(cor) * bg_scale
        cor *= removalmask
        # print('Applied median subtraction and mask')
        return cor

class scaledbg(object):
    """
    data = scale * bg + offset
    frim ImageD11.scale (but with a pixel selection)
    """
    def __init__(self, bgimage, npx = 2048*8):
        order = np.argsort( bgimage.ravel() )
        self.bgimage = bgimage.copy().astype(np.float32)
        self.indx = np.concatenate( (order[:npx], order[-npx:]) )
        self.bgsample = self.bgimage.ravel()[ self.indx ]
        self.lsqmat = np.empty( (2,2), float )
        self.lsqmat[0,0] = np.dot(self.bgsample, self.bgsample)
        self.lsqmat[1,0] = self.lsqmat[0,1] = self.bgsample.sum()
        self.lsqmat[1,1] = len(self.bgsample)
        self.inverse = np.linalg.inv( self.lsqmat )
        self.bgestimage = np.empty_like( self.bgimage )

    def fit(self, other):
        imsample = other.ravel()[ self.indx ]
        rhs = ( np.dot( imsample, self.bgsample), imsample.sum() )
        return np.dot( self.inverse, rhs )

    def estimate(self, other):
        scale, offset = self.fit( other )
        # blas would be better
        np.multiply( self.bgimage, scale, self.bgestimage )
        np.add( self.bgestimage, offset, self.bgestimage )
        return self.bgestimage
    
def init(self):
    # validate input
    # print("# max pixels",self.HOWMANY)
    self.CUT = self.THRESHOLDS[0]
    assert self.CUT >= 0
    for i in range(1, len(self.THRESHOLDS)):
        assert self.THRESHOLDS[i] > self.THRESHOLDS[i - 1]
    #print("# thresholds", str(self.THRESHOLDS))
    if self.MASKFILE is not None:
        self.MASK = 1 - fabio.open(self.MASKFILE).data
        # print("# Opened mask", self.MASKFILE)
    else:
        self.MASK = None
    if self.CORES is None:
        try:
            self.CORES = int(os.environ["SLURM_JOB_CPUS_PER_NODE"]) - 1
        except:
            self.CORES = os.cpu_count() - 1
    self.CORES = max(1, self.CORES)
    # print("# Aiming to use", self.CORES, "processes")
    threshold_value = int(sys.argv[3])
    outh5 = os.path.split(self.HNAME)[-1].replace(".h5", f"_t{threshold_value}_sparse.h5")
    self.OUTNAME = os.path.join(self.OUTPATH, outh5)
    # print("# Output to ", self.OUTNAME)
    try:
        self.BG = scaledbg( fabio.open(self.BACKGROUND).data )
        # print("# Background subtracted from",self.BACKGROUND)
    except:
        self.BG=None

Options.__init__ = init
# each subprocess gets their own options (no need if we fork?)
OPTIONS = Options()


# we will use multiple processes, each with 1 core
# all sub-processes better do this!
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"  # not sure we want/need this here?
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"


@numba.njit
def select(img, mask, row, col, val, cut):
    # Choose the pixels that are > cut and put into sparse arrays
    k = 0
    for s in range(img.shape[0]):
        for f in range(img.shape[1]):
            if img[s, f] > cut:
                if mask[s, f]:  # skip masked
                    continue
                row[k] = s
                col[k] = f
                val[k] = img[s, f]
                k += 1
    return k

@numba.njit
def top_pixels(nnz, row, col, val, howmany, thresholds):
    """
    selects the strongest pixels from a sparse collection
    - THRESHOLDS should be a sorted array of potential cutoff values to try
    that are higher than the original cutoff used to select data
    - howmany is the maximum number of pixels to return
    """
    # quick return if there are already few enough pixels
    if nnz <= howmany:
        return nnz, thresholds[0]
    # histogram of how many pixels are above each threshold
    h = np.zeros(len(thresholds), dtype=np.uint32)
    for k in range(nnz):
        for i, t in enumerate(thresholds):
            if val[k] > t:
                h[i] += 1
            else:
                break
    # choose the one to use. This is the first that is lower than howmany
    tcut = thresholds[-1]
    for n, t in zip(h, thresholds):
        if n < howmany:
            tcut = t
            break
    # now we filter the pixels
    n = 0
    for k in range(nnz):
        if val[k] > tcut:
            row[n] = row[k]
            col[n] = col[k]
            val[n] = val[k]
            n += 1
            if n >= howmany:
                break
    return n, tcut

@functools.lru_cache(maxsize=1)
def get_dset(h5name, dsetname):
    """This avoids re-reading the dataset many times"""
    with h5py.File(h5name, "r") as h5file:
        if dsetname not in h5file:
            raise ValueError(f"No dataset or group named {dsetname} in the file {h5name}")
        dset = h5file[dsetname]
        if isinstance(dset, h5py.Group):
            keys = list(dset.keys())
            raise TypeError(f"Expected a dataset at {dsetname}, but got a group with keys: {keys}")
        return dset

def choose_parallel(args):
    """reads a frame and sends back a sparse frame"""
    h5name, address, frame_num = args

    # Open the HDF5 file for each frame
    with h5py.File(h5name, "r") as h5file:
        if address not in h5file:
            raise ValueError(f"No dataset or group named {address} in the file {h5name}")
        
        dset = h5file[address]
        
        # Check if frame_num is a valid index for the dataset
        if frame_num >= len(dset):
            raise IndexError(f"Frame number {frame_num} out of range for dataset with {len(dset)} frames.")
        
        frm = dset[frame_num]
        cor = OPTIONS.correct(frm)
        
    return frame_num, dosegment(cor)


def dosegment( frm ):
    if dosegment.cache is None:
        # cache the mallocs on this function. Should be one per process
        row = np.empty(frm.size, np.uint16)
        col = np.empty(frm.size, np.uint16)
        val = np.empty(frm.size, frm.dtype)
        if OPTIONS.MASK is None:
            msk = np.zeros( frm.shape, bool)
        else:
            msk = OPTIONS.MASK
        dosegment.cache = row, col, val, msk
    else:
        row, col, val, msk = dosegment.cache
    nnz = select(frm, msk, row, col, val, OPTIONS.CUT)
    if nnz == 0:
        sf = None
    else:
        if nnz > OPTIONS.HOWMANY:
            nnz, tcut = top_pixels(nnz, row, col, val, OPTIONS.HOWMANY,  OPTIONS.THRESHOLDS)
            #print("t=%d"%(tcut),end=' ')
        # Now get rid of the single pixel 'peaks'
        #   (for the mallocs, data is copied here)
        s = sparseframe.sparse_frame(row[:nnz].copy(), col[:nnz].copy(), frm.shape)
        s.set_pixels("intensity", val[:nnz].copy())
        if OPTIONS.PIXELS_IN_SPOT <= 1:
            sf = s
        else:
            # label them according to the connected objects
            sparseframe.sparse_connected_pixels(
                s, threshold=OPTIONS.CUT, data_name="intensity", label_name="cp"
            )
            # only keep spots with more than 3 pixels ...
            mom = sparseframe.sparse_moments(
                s, intensity_name="intensity", labels_name="cp"
            )
            npx = mom[:, cImageD11.s2D_1]
            pxcounts = npx[s.pixels["cp"] - 1]
            pxmsk = pxcounts >= OPTIONS.PIXELS_IN_SPOT
            if pxmsk.sum() == 0:
                sf = None
            else:
                sf = s.mask(pxmsk)
    return sf

dosegment.cache = None  # cache for malloc per process


def segment_scans( fname,
                   scans,
                   outname,
                   mypool,
                   scanmotors=OPTIONS.SCANMOTORS,
                   headermotors=OPTIONS.HEADERMOTORS,
                   detector=OPTIONS.DETECTORNAME ):
    """Does segmentation on a series of scans in hdf files:
    fname : input hdf file
    scans : input groups in the hdf files [h[s] for s in scans] will be treated
    outname : output hdf file to put the results into
    sparsify_func : function to select which pixels to keep
    headermotors : things to copy from input instrument/positioners to output
    scanmotors : datasets to copy from input measurement/xxx to output
    """
    opts = {
        "chunks": (10000,),
        "maxshape": (None,),
        "compression": "lzf",
        "shuffle": True,
    }
    ndone = 0
    with h5py.File(outname, "a") as hout:
        for scan in scans:
            if scan.endswith(".2"):  # for fscans
                continue
            with h5py.File(fname, "r") as hin:
                hout.attrs["h5input"] = fname
                gin = hin[scan]
                '''
                bad = 1
                if "title" not in hin[scan]:
                    print(scan,"missing title, skipping")
                    continue
                if "scan_data" not in hin[scan]:
                #if "measurement" not in hin[scan]:
                    print(scan,"missing measurement group, skipping")
                    continue
                if detector not in hin[scan]['measurement']:
                    print(scan,"missing measurement/%s, skipping"%(detector))
                    continue
                title = hin[scan]["title"][()]
                print("# ", scan, title)
                print("# time now", time.ctime(), "\n#", end=" ")
                for scantype in OPTIONS.SCANTYPES:
                    if title.find(scantype) >= 0:
                        bad = 0
                if ("measurement" in gin) and (detector not in gin["measurement"]):
                    bad += 1
                if bad:
                    print("# SKIPPING BAD SCAN", scan)
                    continue
                g = hout.create_group(scan)
                gm = g.create_group("measurement")
                for m in scanmotors:  # vary : many
                    if m in gin["measurement"]: #fix by add ["positioners"], Axel H. 24 feb 2022.
                        gm.create_dataset(m, data=gin["measurement"][m][:])
                        print("adding scanmotor: ", list(g["measurement"]))
                    else:
                        raise ValueError(str(m)+' is not in measurement'+ str( list(gin["measurement"]) ) )
                gip = g.create_group("instrument/positioners")
                for m in headermotors:  # fixed : scalar
                    if "instrument/positioners/%s"%(m) in gin:
                        gip.create_dataset(m, data=gin["instrument/positioners"][m][()])
                try:
                    frms = gin["measurement"][detector]
                except:
                    print(list(gin))
                    print(list(gin["measurement"]))
                    print(detector)
                    raise
                '''
                g = hout.create_group(scan)
                gm = g.create_group("measurement")
                # tmp = np.array(range(3600))/10
                length = int(sys.argv[4])  # get number of frames from argv, this is for omega interval
                # tmp = np.array(range(length))/10
                interval = 360.0 / length
                tmp = np.array(range(length)) * interval
                tmp2 = np.array([0.])
                gm.create_dataset("diffrz",data=tmp[:])
                gm.create_dataset("diffrz_center",data=tmp[:])
                # print(np.array(g['measurement/diffrz']))
                gip = g.create_group("instrument/positioners")
                gip.create_dataset("diffty",data=tmp2[0])
                hout = h5py.File(fname,'r')
                # print(hout.keys())
                # print(fname)
                #frms = hout['images']
                if 'images' in hout:
                    frms = hout['images']
                elif 'imageseries/images' in hout:
                    frms = hout['imageseries/images']
                else:
                    raise ValueError("Neither 'images' nor 'imageseries/images' paths found in the HDF5 file.")
                frms = np.swapaxes(frms,1,2)
                #frms = hout['flyscan_00001/scan_data/images']
                row = g.create_dataset("row", (1,), dtype=np.uint16, **opts)
                col = g.create_dataset("col", (1,), dtype=np.uint16, **opts)
                # can go over 65535 frames in a scan
                num = g.create_dataset("frame", (1,), dtype=np.uint32, **opts)
                sig = g.create_dataset("intensity", (1,), dtype=frms.dtype, **opts)
                nnz = g.create_dataset("nnz", (frms.shape[0],), dtype=np.uint32)
                g.attrs["itype"] = np.dtype(np.uint16).name
                g.attrs["nframes"] = frms.shape[0]
                g.attrs["shape0"] = frms.shape[1]
                g.attrs["shape1"] = frms.shape[2]
                # print("shape="+str(frms.shape[0]))
                # print("gattr shape0="+str(g.attrs["shape0"]))
                npx = 0
                if scan == "imageseries":
                    address = scan + "/images"
                else:
                    address = scan
                #address = scan + "/measurement/" + detector
                nimg = frms.shape[0]
                args = [(fname, address, i) for i in range(nimg)]
            # CLOSE the input file here
            # max chunksize is for load balancing. Originally tuned to be 7200/40/8 == 22
            max_chunksize = 64
            chunksize = min( max(1, len(args) // OPTIONS.CORES // 8), max_chunksize )
            # set the timeout to be 1 second per frame plus a minute. If we can't segment at
            # 1 f.p.s better to give up.
            timeout = len(args) + 60
            for i, spf in mypool.map(
                choose_parallel, args, chunksize=chunksize, timeout=timeout
            ):
                if i % 500 == 0:
                    if spf is None:
                        # print("%4d 0" % (i), end=",")
                        pass
                    else:
                        # print("%4d %d" % (i, spf.nnz), end=",")
                        pass
                    sys.stdout.flush()
                if spf is None:
                    nnz[i] = 0
                    continue
                if spf.nnz + npx > len(row):
                    row.resize(spf.nnz + npx, axis=0)
                    col.resize(spf.nnz + npx, axis=0)
                    sig.resize(spf.nnz + npx, axis=0)
                    num.resize(spf.nnz + npx, axis=0)
                row[npx:] = spf.row[:]
                col[npx:] = spf.col[:]
                sig[npx:] = spf.pixels["intensity"]
                num[npx:] = i
                nnz[i] = spf.nnz
                npx += spf.nnz
            ndone += nimg
            # print("\n# Done", scan, nimg, ndone)
    return ndone

    # the output file should be flushed and closed when this returns


def main(hname, outname, mypool):
    # Remove old *sparse.h5 file if exit
    if os.path.exists(outname):
        os.remove(outname)
    # read all the scans : just the master process
    # print("hname = ", repr(hname))
    # print("outname = ", repr(outname))
    with h5py.File(hname, "r") as h:
        scans = list(h["/"])
        # print("scans = ", repr(scans))
    # print("#", time.ctime())
    start = time.time()
    # print("outname= "+str(outname))
    nfrm = segment_scans(hname, scans, outname, mypool)
    end = time.time()
    # print("#", time.ctime())
    #print("# Elapsed", end - start, "/s,   f.p.s %.2f" % (nfrm / (end - start)))


if __name__ == "__main__":
    with concurrent.futures.ProcessPoolExecutor(max_workers=OPTIONS.CORES) as thepool:
        main(OPTIONS.HNAME, OPTIONS.OUTNAME, thepool)

