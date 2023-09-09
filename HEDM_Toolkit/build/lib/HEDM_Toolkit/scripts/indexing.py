import sys, random
import os
from ImageD11.grid_index_parallel import grid_index_parallel

if __name__=="__main__":
    # You need this idiom to use multiprocessing on windows (script is imported again)
    
    symmetry = sys.argv[9]
    ring1 = list(map(int, sys.argv[10].split(',')))
    ring2 = list(map(int, sys.argv[11].split(',')))
    
    gridpars = {
        'DSTOL' : 0.004,
        'OMEGAFLOAT' : 0.13,
        'COSTOL' : 0.002,
        'NPKS' : int(sys.argv[4]),
        'TOLSEQ' : [float(sys.argv[6]), float(sys.argv[7])],
        'SYMMETRY' : symmetry,
        'RING1'  : ring1,
        'RING2' : ring2,
        'NUL' : True,
        'FITPOS' : True,
        'tolangle' : 1.0,
        'toldist' : int(sys.argv[8]),
        'NPROC' : None,
        'NTHREAD' : 1,
    }

    try:
        gridpars['NUNIQ'] = int(sys.argv[5])
    except:
        pass
    # grid to search
    r = 800    # radius of cylinder
    r2 = 500   # Beam height
    translations = [(t_x, t_y, t_z)
        for t_x in range(-r, r+1, 200)
        for t_y in range(-r, r+1, 200)
        for t_z in range(-r2, r2+1, 200) ]
    # Cylinder:
    translations = [( x,y,z) for (x,y,z) in translations if (x*x+y*y)< r*r ]
    #
    random.seed(42) # reproducible
    random.shuffle(translations)

    fltfile = sys.argv[1]
    parfile = sys.argv[2]
    tmp = sys.argv[3]
    
    grid_index_parallel( fltfile, parfile, tmp, gridpars, translations)