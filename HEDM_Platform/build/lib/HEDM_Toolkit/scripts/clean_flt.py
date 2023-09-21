import ImageD11.columnfile
import ImageD11.indexing
import scipy.spatial, scipy.sparse
import numpy as np, pylab as pl, pprint
import sys
import os

baseDir = "%s" % sys.argv[1]
str1 = "%s" % sys.argv[2]
str2 = "%s" % sys.argv[3]
str3 = "%s" % sys.argv[4]
merge_tol = float(sys.argv[5])

#str1 = "ottawa15_diff1/ottawa15_diff1_scan1/ottawa15_diff1_scan1.flt"
#str2 = "fitted_soleil_221029.par"
c = ImageD11.columnfile.columnfile(os.path.join(baseDir, str1))

#c = ImageD11.columnfile.columnfile(str1)
c.parameters.loadparameters(str2)

maxid = 96486
#pl.plot(c.omega,'.')
#pl.figure()
x=np.arange(20000,20150)
#x1 = range(maxid)
#pl.plot(x1,c.tth[0:maxid],'.')
#pl.plot(x1,c.tth[maxid:maxid*2],'+')
#pl.show()

# MUST MODIFY THIS FOR NEW DATA
maxid = int(round(float(c.omega.size/13))) # for diff1-scan1 TODO: fix this to make it automatic
c = c.copyrows(range(0,maxid))

# Sort by omega
c.sortby('omega')
omegavals = np.unique(c.omega)
omegavals.sort()

# try to merge the data over omega
n = [ (c.omega == o).sum() for o in omegavals]
p = np.cumsum( np.concatenate(([0,], n ) ) )
#pl.figure()
#pl.plot(n)
#pl.show()
# make a KDTree for each frame (wastes a bit of memory, but easier for sorting later)
trees = [ scipy.spatial.cKDTree( np.transpose( (c.sc[p[i]:p[i+1]], c.fc[p[i]:p[i+1]]) ) )
          for i in range(len(n)) ]
# peaks that overlap, k : 0 -> npks == len(s|f|o)
# diagonal
krow = list(range(c.nrows))
kcol = list(range(c.nrows))

for i in range(1,len(n)):  # match these to previous
    tlo = trees[i-1]
    thi = trees[i]
    # 1.6 is how close centers should be to overlap  Tian: 1.6--merge_tol
    lol = trees[i-1].query_ball_tree( trees[i], r = merge_tol )
    for srcpk, destpks in enumerate(lol):  # dest is strictly higher than src
        for destpk in destpks:
            krow.append( srcpk + p[i-1] )
            kcol.append( destpk + p[i] )
csr = scipy.sparse.csr_matrix( ( np.ones(len(krow), dtype=bool),
                               (kcol, krow) ), shape=(c.nrows,c.nrows) )
    # connected components == find all overlapping peaks
ncomp, labels= scipy.sparse.csgraph.connected_components( csr,
                                         directed=False, return_labels=True )
print(ncomp, labels.shape, c.nrows)

# now merge peaks across labels
c.titles
todo = [t for t in c.titles]
merged = {}
# things that are a simple sum
for item in 'Number_of_pixels', 'sum_intensity':
    data = c.getcolumn(item)
    merged[item] = np.bincount( labels, minlength=ncomp, weights=data)
# things that are an intensity weighted sum
ws = c.sum_intensity
wn = merged['sum_intensity']
for item in 'f_raw','s_raw','omega','sc','fc': #  'dty': # dty is not used so I am skipping it
    data = c.getcolumn(item) * ws
    wsum = np.bincount( labels, minlength=ncomp, weights=data)
    merged[item] = wsum / wn

mc = ImageD11.columnfile.colfile_from_dict( {t:merged[t] for t in c.titles if t in merged} )
mc.parameters.loadparameters(str2)
mc.updateGeometry()
mc.nrows, c.nrows
#pl.figure()
#pl.plot(c.omega, c.eta,'.')
#pl.plot(mc.omega, mc.eta,'+', ms=10)
#pl.xlim(0,10)
#pl.show()

# Plot things
#pl.plot(c.tth, c.eta,',')
#pl.plot(mc.tth, mc.eta,',')
#pl.show()

# Save things
mc.writefile(os.path.join(baseDir, str3))
