import argparse
import logging
import sys

from ImageD11 import ImageD11options, refinegrains
from ImageD11.indexing import readubis, write_ubi_file


def makemap(options):
    tthr = options.tthrange or (0., 180.)
    print("Using tthrange", tthr)

    func = getattr(refinegrains, options.latticesymmetry)
    o = refinegrains.refinegrains(intensity_tth_range=tthr,
                                  latticesymmetry=func,
                                  OmFloat=options.omega_float,
                                  OmSlop=options.omega_slop)

    o.loadparameters(options.parfile)
    o.loadfiltered(options.fltfile)
    o.readubis(options.ubifile)

    if options.symmetry != "triclinic":
        o.makeuniq(options.symmetry)

    o.tolerance = float(options.tol)
    o.generate_grains()
    o.refinepositions()
    o.savegrains(options.newubifile, sort_npks=options.sort_npks)

    col = o.scandata[options.fltfile].writefile(options.fltfile + ".new")

    if hasattr(options, "newfltfile") and options.newfltfile is not None:
        o.assignlabels()
        col = o.scandata[options.fltfile].copy()
        col.filter(col.labels < -0.5)
        col.writefile(options.newfltfile)


def get_options():
    parser = refinegrains.get_options(argparse.ArgumentParser())
    parser.add_argument("--no_sort", action="store_false",
                        dest="sort_npks", default=True,
                        help="Sort grains by number of peaks indexed")
    parser.add_argument("--tthrange", action="append",
                        dest="tthrange", type=float,
                        help="Two theta range for getting median intensity")
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format='%(levelname)-8s : %(message)s')
    parser = get_options()
    options = parser.parse_args()

    required_args = ["parfile", "ubifile", "newubifile", "fltfile"]
    missing_args = [arg for arg in required_args if getattr(options, arg) is None]
    
    if missing_args:
        parser.print_help()
        logging.error(f"Missing options: {', '.join(missing_args)}")
        sys.exit()

    try:
        makemap(options)
    except Exception as e:
        parser.print_help()
        logging.error(f"An error occurred: {str(e)}")
 