import sys, random
import os
from ImageD11.grid_index_parallel import grid_index_parallel


if __name__=="__main__":
    # You need this idiom to use multiprocessing on windows (script is imported again)
    gridpars = {
        'DSTOL' : 0.004, #0.003, # tol in 1/d
        'OMEGAFLOAT' : 0.13, # 0.025
        'COSTOL' : 0.002, # was 0.002
        'NPKS' : int(  sys.argv[4] ),
        'TOLSEQ' : [float(sys.argv[6]), float(sys.argv[7])], # first item was 0.02, 0.015, 0.01 #[0] is hkl tolerance. Generally reduce to reduce grains
        'SYMMETRY' : "hexagonal",
        'RING1'  : [1,2], #note: rings start from 0!
        'RING2' : [1,2,4,5,6,7,8,9,11,12,14],
        'NUL' : True,
        'FITPOS' : True, # Does a preliminary fitting of the grains
        'tolangle' : 1.0, #0.25, # Turn down to increase grains
        'toldist' : int(sys.argv[8]), #100., # Turn down to increase grains
        'NPROC' : 48, # guess from cpu_count
        'NTHREAD' : 1 ,
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