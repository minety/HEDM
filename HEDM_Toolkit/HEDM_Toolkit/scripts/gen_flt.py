import sys, os
# os.environ['OMP_NUM_THREADS']='1'  # use multiprocessing not openmp
import time
import h5py, numpy as np, numba, fabio, multiprocessing #, pylab as plt
from ImageD11 import unitcell
from ImageD11 import sparseframe, cImageD11, columnfile, transform
import fast_histogram
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from ImageD11.sparseframe import sparse_frame
fnum  = 1 #10

def moments( bfrm ):
    I = bfrm[:, cImageD11.s2D_I]
    s = bfrm[:, cImageD11.s2D_sI] / I
    f = bfrm[:, cImageD11.s2D_fI] / I
    n = bfrm[:, cImageD11.s2D_1]
    a = I / n
    return s, f, a, n

def newpks(hf, scans=None, npixels=1, monitor=None, monval=1e3, ):
    """ do a peak search in a sparse frame series """
    titles = 's_raw f_raw avg_intensity Number_of_pixels sum_intensity omega dty'.split()
    c = {}
    for name in titles:
        c[name] = []
    # using the python file open overcomes some threading crap
    with h5py.File(open(hf,"rb"),'r') as hin:
        # scan numbers
        if scans is None:
            scans = list(hin['/'])
        for scan in scans:
            scan = 'images'
            gin = hin[scan]
            shape = gin.attrs['shape0'], gin.attrs['shape1']
            # import pdb; pdb.set_trace()
            omega = gin['measurement/diffrz'][:]
            difty = gin['instrument/positioners/diffty'][()]
            row = gin['row'][()]
            col = gin['col'][()]
            sig = gin['intensity'][()]
            if monitor is not None:
                mon = monval/gin[monitor][()]
            ipt = np.cumsum( gin['nnz'][:] )
            iprev = 0
            for k,nnz in enumerate( gin['nnz'][()] ) :
                inext = iprev + nnz
                if nnz == 0:
                    continue
                f = sparseframe.sparse_frame( row[iprev:inext],
                                              col[iprev:inext], shape )
                f.set_pixels("intensity", sig[iprev:inext] )
                sparseframe.sparse_connected_pixels(f, threshold=0.1)
                pks = sparseframe.sparse_moments(f, "intensity", "connectedpixels")
                #sparseframe.sparse_localmax(f)
                #pks = sparseframe.sparse_moments(f, "intensity", "localmax")
                s,f,a,n = moments( pks )
                m = n > npixels
                c['s_raw'].append( s[m] )
                c['f_raw'].append( f[m] )
                if monitor is not None:
                    c['avg_intensity'].append( a[m]*mon[k] )
                    c['sum_intensity'].append( a[m]*n[m]*mon[k] )
                else:
                    c['avg_intensity'].append( a[m] )
                    c['sum_intensity'].append( a[m]*n[m] )
                c['Number_of_pixels'].append( n[m] )
                npk = m.sum()
                c['omega' ].append( np.full( npk , omega[k] ) )
                c['dty'].append( np.full( npk , difty ) )
                iprev = inext
    for t in titles:
        c[t] = np.concatenate( c[t] )
    return c

def pfun( *args ):
#    print(args[0])
    s, h5name, npixels = args[0]
    return (s, newpks( h5name, [s,], npixels = npixels))

outname = "%s" % sys.argv[1]  #input h5 file
sparsename = "%s" % sys.argv[2]
filtername = "%s" % sys.argv[3]
par_file = "%s" % sys.argv[4]
dx_file = "%s" % sys.argv[5]
dy_file = "%s" % sys.argv[6] 

def sortedscans( hfo ):
    return [sc for _,sc in sorted([(float(s),s) for s in list(hfo)])]

#with h5py.File(outname,'r') as h:
    #im = h['flyscan_00001/scan_data/orca_image']
    #im3 = np.swapaxes(im,1,2)
    #bg_frames = np.median(im3[0::50,:,:],axis=0)
    #scans = sortedscans( h )
    #for scan in scans:
    #    r = h[scan]['measurement/diffrz'][()]
    #    print(scan,h[scan]['instrument/positioners/diffty'][()], r[0])
    #bg_frames = h[scan]['measurement/frelon3'][::100]

with h5py.File(outname, 'r') as hin:
    #frm = hin[scan]['measurement/frelon3'][fnum]
    im = hin['images']
    #im3 = np.swapaxes(im,1,2)
    frm = im[fnum,:,:]

with h5py.File(sparsename,"r") as hin:
    scan='images'
    print(list(hin[scan]), dict(hin[scan].attrs))
    print(list(hin[scan]['instrument']))
    f = fnum
    nnz = hin[scan]['nnz'][:]
    ipt = np.cumsum(nnz)
    # num_frames = len(hin[scan]['frame'])

    # if num_frames == 2:
    #     # Adjust the logic for 2 frames
    #     # For example, if you know `f` should be 1 for the second frame, set it manually:
    #     f = 1
    #     print(f, ipt[f-1], hin[scan]['frame'][ipt[f-1]])
    # else:
    #     # The original logic for more frames
    #     print(f, ipt[f-1], ipt[f-2], hin[scan]['frame'][ipt[f-1]], hin[scan]['frame'][ipt[f]])

    s = ipt[f-1]
    e = ipt[f]
    g = hin[scan]
    sh = g.attrs['shape0'],g.attrs['shape1']
    spf = sparse_frame( g['row'][s:e], g['col'][s:e], shape = (g.attrs['shape0'],g.attrs['shape1']),
                       pixels={'intensity': g['intensity'][s:e]})
    seg = spf.to_dense('intensity')

class spatial(object):
    def __init__(self,
                 dxfile=dx_file,
                 dyfile=dy_file,):
        import fabio
        self.dx = fabio.open(dxfile).data
        self.dy = fabio.open(dyfile).data
    def __call__(self, pks):
        si = np.round(pks['s_raw']).astype(int)
        fi = np.round(pks['f_raw']).astype(int)
        pks['sc'] = self.dy[ si, fi ] + pks['s_raw']
        pks['fc'] = self.dx[ si, fi ] + pks['f_raw']
        pks['sc'] = pks['s_raw']
        pks['fc'] = pks['f_raw']
        return pks

def tocolf(pks):
    titles = list(pks.keys())
    colf = columnfile.newcolumnfile( titles=titles )
    nrows = len(pks[titles[0]])
    colf.nrows = nrows
    colf.set_bigarray( [ pks[t] for t in titles ] )
    return colf

spat = spatial(  )

sparsename_load = sparsename

# Open hdf5 file of sparse peaks
with h5py.File(sparsename_load,'r') as h:
    #scans_load = sortedscans( h )
    #for scan in scans_load:
    scans = 'images'
    scans_load = 'images'
    #r = h[scan]['measurement/diffrz'][()]
    #print(scan,h[scan]['instrument/positioners/diffty'][()], r[0])

print('scans_load='+scans_load)
pks_load = newpks(sparsename_load, scans=scans_load, npixels=2, monitor=None)

# Save flt filtered peaks file
pks_load.keys(), 12.3985/42.5
pks_load = spat(pks_load)
colf_load = tocolf(pks_load)
colf_load.filter(colf_load.Number_of_pixels>5)
colf_load.parameters.loadparameters(par_file)
colf_load.updateGeometry()
colf_load.filter(colf_load.Number_of_pixels*np.exp(0.4*colf_load.ds**2)>40)
#plt.figure()
#plt.plot(colf_load_02.tth, colf_load_02.eta,'.',alpha=0.3)

# Get number of lines in filtered peaks file
print('num spots = '+str(colf_load.nrows))
uc_load = unitcell.unitcell_from_parameters(colf_load.parameters)
uc_load.makerings(colf_load.ds.max())
s2 = colf_load.ds
colf_load.filter( colf_load.ds < 0.75)
la = colf_load.copy()
la.filter( colf_load.ds < 0.75)
la.writefile(filtername)
#m = abs(colf_load.ds - 0.23) < 0.01
#plt.figure()
#plt.plot(180-colf_load.eta, 180+colf_load.omega, 'o', alpha=0.2)

# Set the minimal things to save
minimal = { k:colf_load[k] for k in ['sc','fc','omega','dty','sum_intensity','Number_of_pixels']}
colf_load = minimal

# Create random number
def rnd( a, p, t):
    if p == 0:
        return a.astype(t)
    # hack to try to get floats to compress better. Round to nearest integer p
    a = (np.round(a*pow(2,p)).astype(np.int).astype(t))/pow(2,p)
    return a
# # Save sparse peaks hdf5 file
# with h5py.File(pksname,'w') as hout:
#     g = hout.create_group('peaks')
#     g.attrs['ImageD11_type'] = 'peaks'
#     for name,b,t in ( ('sc',8, np.float32),
#                  ('fc',8, np.float32),
#                  ('omega',8, np.float32),
#                  ('dty',8, np.float32),
#                  ('sum_intensity',2, np.float32),
#                  ('Number_of_pixels',0, np.uint32)):
#         a = rnd(colf_load[name],b,t)
#         print(name, 'chunks', a.size/pow(2,11))
#         g.create_dataset( name,
#                           data = a,
#                           dtype = t,
#                           compression = 'lzf',
#                           shuffle = True,
#                           chunks = ( pow(2,11) ,), #  32 kB is
#                         )
