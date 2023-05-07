import numpy as np
from hexrd import imageseries
from hexrd.imageseries.omega import OmegaWedges

def bgsub4hdf5(h5input, h5output, num_images, omega, numframes, bg_pct, bg_nf):

    ims = imageseries.open(
        h5input,
        format='hdf5',
        path='/',
        dataname='images'
    )

    # Input of omega meta data
    nf = numframes  #720
    omega = omega
    omw = OmegaWedges(nf)
    omw.addwedge(0, nf*omega, nf) 
    ims.metadata['omega'] = omw.omegas

    # Make dark image from first 100 frames
    pct = bg_pct
    nf_to_use = bg_nf
    dark = imageseries.stats.percentile(ims, pct, nf_to_use)
    # np.save(DarkFile, dark)

    # Now, apply the processing options
    ProcessedIS = imageseries.process.ProcessedImageSeries
    ops = [('dark', dark), ('flip', 'h')] # None, 'h', 'v', etc.
    pimgs = ProcessedIS(ims, ops)

    # Save the processed imageseries in hdf5 format
    imageseries.write(pimgs, h5output,'hdf5', path='/imageseries')

if __name__ == "__main__":
    h5input = '/Users/yetian/Desktop/Ryan_test_data/APS_2023Feb/nf_test/nugget1_nf_int_det1.hdf5' 
    print(h5input)
    h5output = '/Users/yetian/Desktop/Ryan_test_data/APS_2023Feb/nf_test/nugget1_nf_int_det1_50bg.h5'
    print(h5output)
    num_images = 180 # number of frames in total in hdf5 file
    omega = 1 # omega rotation angle in degree
    numframes = 180 # number of frames to export may be different with num_images which include some empty images
    bg_pct = 50 # background to subtract in percentile
    bg_nf = 90 # number of frames to use to generate dark field image               
    bgsub4hdf5(h5input, h5output, num_images, omega, numframes, bg_pct, bg_nf)






