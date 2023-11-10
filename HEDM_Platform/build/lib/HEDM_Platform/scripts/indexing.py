import sys
import random
import os
from ImageD11.grid_index_parallel import grid_index_parallel

def generate_translations(r, r2):
    """ Generate cylinder translation grid """
    translations = [(t_x, t_y, t_z)
                    for t_x in range(-r, r+1, 200)
                    for t_y in range(-r, r+1, 200)
                    for t_z in range(-r2, r2+1, 200)]
    
    # Filter for cylinder:
    return [(x, y, z) for (x, y, z) in translations if (x*x + y*y) < r*r]

if __name__ == "__main__":
    fltfile, parfile, tmp, npks, _, tolseq1, tolseq2, toldist, symmetry, ring1_str, ring2_str, tolangle, r, h, nproc = sys.argv[1:]
    
    ring1 = list(map(int, ring1_str.split(',')))
    ring2 = list(map(int, ring2_str.split(',')))
    r = int(h)/2     # Beam height/2
    r2 = int(r)

    gridpars = {
        'DSTOL': 0.004,
        'OMEGAFLOAT': 0.13,
        'COSTOL': 0.002,
        'NPKS': int(npks),
        'TOLSEQ': [float(tolseq1), float(tolseq2)],
        'SYMMETRY': symmetry,
        'RING1': ring1,
        'RING2': ring2,
        'NUL': True,
        'FITPOS': True,
        'tolangle': float(tolangle),
        'toldist': int(toldist),
        'NPROC': None if nproc.lower() == 'none' else int(nproc),
        'NTHREAD': 1,
        'NUNIQ': int(sys.argv[5])
    }

    translations = generate_translations(int(r), r2)

    # Shuffle for reproducibility
    random.seed(42)
    random.shuffle(translations)

    grid_index_parallel(fltfile, parfile, tmp, gridpars, translations)
