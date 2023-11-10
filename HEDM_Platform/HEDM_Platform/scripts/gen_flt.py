import h5py
import numpy as np
import fabio
from ImageD11 import unitcell, sparseframe, cImageD11, columnfile, transform
from concurrent.futures import ProcessPoolExecutor
from ImageD11.sparseframe import sparse_frame
import sys
np.bool = bool

def moments(bfrm):
    I = bfrm[:, cImageD11.s2D_I]
    s = bfrm[:, cImageD11.s2D_sI] / I
    f = bfrm[:, cImageD11.s2D_fI] / I
    n = bfrm[:, cImageD11.s2D_1]
    a = I / n
    return s, f, a, n


def get_scan_path(hin):
    if 'images' in hin:
        return 'images'
    elif 'imageseries' in hin:
        return 'imageseries'
    else:
        raise ValueError("Neither 'images' nor 'imageseries' paths found in the HDF5 file.")


def newpks(hf, scans=None, npixels=1, monitor=None, monval=1e3):
    titles = 's_raw f_raw avg_intensity Number_of_pixels sum_intensity omega dty'.split()
    c = {name: [] for name in titles}
    with h5py.File(hf, 'r') as hin:
        if scans is None:
            scans = list(hin['/'])
        for scan in scans:
            scan_path = get_scan_path(hin)
            gin = hin[scan_path]
            omega = gin['measurement/diffrz'][:]
            difty = gin['instrument/positioners/diffty'][()]
            row, col, sig = gin['row'][()], gin['col'][()], gin['intensity'][()]
            ipt = np.cumsum(gin['nnz'][:])
            iprev = 0
            for k, nnz in enumerate(gin['nnz'][()]):
                inext = iprev + nnz
                if nnz == 0:
                    continue
                f = sparse_frame(row[iprev:inext], col[iprev:inext], shape=(gin.attrs['shape0'], gin.attrs['shape1']))
                f.set_pixels("intensity", sig[iprev:inext])
                sparseframe.sparse_connected_pixels(f, threshold=0.1)
                pks = sparseframe.sparse_moments(f, "intensity", "connectedpixels")
                s, f, a, n = moments(pks)
                m = n > npixels
                c['s_raw'].append(s[m])
                c['f_raw'].append(f[m])
                c['avg_intensity'].append(a[m] * (monval / gin[monitor][()] if monitor else 1))
                c['sum_intensity'].append(a[m] * n[m] * (monval / gin[monitor][()] if monitor else 1))
                c['Number_of_pixels'].append(n[m])
                npk = m.sum()
                c['omega'].append(np.full(npk, omega[k]))
                c['dty'].append(np.full(npk, difty))
                iprev = inext
    for t in titles:
        c[t] = np.concatenate(c[t])
    return c
    
class Spatial:
    def __call__(self, pks):
        pks['sc'] = pks['s_raw']
        pks['fc'] = pks['f_raw']
        return pks

def tocolf(pks):
    colf = columnfile.newcolumnfile(titles=list(pks.keys()))
    nrows = len(pks[next(iter(pks.keys()))])
    colf.nrows = nrows
    colf.set_bigarray([pks[t] for t in pks.keys()])
    return colf


if __name__ == "__main__":
    outname = sys.argv[1]
    sparsename = sys.argv[2]
    filtername = sys.argv[3]
    par_file = sys.argv[4]

    with h5py.File(outname, 'r') as hin:
        scan_path = get_scan_path(hin)
        if scan_path == 'images':
            im_dataset = hin[scan_path]
        else:
            im_dataset = hin[scan_path + '/images']
        frm = im_dataset[1, :, :]

    spat = Spatial()
    sparsename_load = sparsename

    with h5py.File(sparsename_load, 'r') as hin:
        scan_path = get_scan_path(hin)
        scans = scan_path

    pks_load = newpks(sparsename_load, scans=scans, npixels=2)
    pks_load = spat(pks_load)
    colf_load = tocolf(pks_load)
    colf_load.filter(colf_load.Number_of_pixels > 5)
    colf_load.parameters.loadparameters(par_file)
    colf_load.updateGeometry()
    colf_load.filter(colf_load.Number_of_pixels * np.exp(0.4 * colf_load.ds ** 2) > 40)
    # Get number of lines in filtered peaks file
    print('num spots = '+str(colf_load.nrows)) 

    la = colf_load.copy()
    la.filter(colf_load.ds < 0.75)
    la.writefile(filtername)
