import ImageD11.columnfile
import ImageD11.indexing
import scipy.spatial, scipy.sparse
import numpy as np
import sys
import os
np.bool = bool

baseDir = sys.argv[1]
str1 = sys.argv[2]
str2 = sys.argv[3]
str3 = sys.argv[4]
merge_tol = float(sys.argv[5])

c = ImageD11.columnfile.columnfile(os.path.join(baseDir, str1))
c.parameters.loadparameters(str2)

# Determine maxid based on omega size
maxid = int(round(float(c.omega.size/13)))
c = c.copyrows(range(0, maxid))

# Sort by omega and get unique omega values
c.sortby('omega')
omegavals = np.unique(c.omega)
omegavals.sort()

n = [(c.omega == o).sum() for o in omegavals]
p = np.cumsum(np.concatenate(([0,], n)))

# Create a KDTree for each frame
trees = [scipy.spatial.cKDTree(np.transpose((c.sc[p[i]:p[i+1]], c.fc[p[i]:p[i+1]])))
         for i, _ in enumerate(omegavals[:-1])]

# Create adjacency matrix for overlapping peaks
krow = list(range(c.nrows))
kcol = list(range(c.nrows))
for i, tree in enumerate(trees[:-1]):
    matches = tree.query_ball_tree(trees[i+1], r=merge_tol)
    for srcpk, destpks in enumerate(matches):
        for destpk in destpks:
            krow.append(srcpk + omegavals[i])
            kcol.append(destpk + omegavals[i+1])

csr = scipy.sparse.csr_matrix((np.ones(len(krow), dtype=bool), (kcol, krow)), shape=(c.nrows, c.nrows))
ncomp, labels = scipy.sparse.csgraph.connected_components(csr, directed=False, return_labels=True)

# Merge peaks across labels based on the connected components
todo = [t for t in c.titles]
merged = {}

# Weighted sum calculations
ws = c.sum_intensity
for item in ['f_raw', 's_raw', 'omega', 'sc', 'fc']:
    data = c.getcolumn(item) * ws
    wsum = np.bincount(labels, minlength=ncomp, weights=data)
    merged[item] = wsum / np.bincount(labels, minlength=ncomp, weights=ws)

mc = ImageD11.columnfile.colfile_from_dict({t: merged[t] for t in c.titles if t in merged})
mc.parameters.loadparameters(str2)
mc.updateGeometry()

# Save results
mc.writefile(os.path.join(baseDir, str3))
