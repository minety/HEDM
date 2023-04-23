import numpy as np
from hexrd import imageseries
from hexrd.imageseries.omega import OmegaWedges

def make_meta():
    return {'testing': '1,2,3'}

def ge2npz(gefile, npzfile, mypath, num_images, omega, numframes, bg_pct, bg_nf):

    # width and height for the images (Default is 2048*2048 for most beamlines)
    width = 2048
    height = 2048

    # number of frames including the empty/bad frames
    num_images = num_images

    # size of each image
    image_size = width * height * 2  # Print if needed

    # Read raw data from ge5 file
    with open(gefile, 'rb') as f:
        # 8192 was found to be the header (skip this part) APS sector1
        f.read(8192)

        # Read all the images as numpy array
        data = np.fromfile(f, dtype=np.uint16)

    # Convert all binary data to an array of images.
    image_data = np.reshape(data, (num_images, height, width))
    ims = imageseries.open(None, 'array', data=image_data, meta=make_meta())

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

    # Save the processed imageseries in npz format
    print("writing npz file (may take a while): %s" % npzfile)
    imageseries.write(pimgs, npzfile, 'frame-cache', path=mypath, threshold=5, cache_file=npzfile) # threshold=5 default


if __name__ == "__main__":
    gefile = '/Users/yetian/Dropbox/My Mac (Ye’s MacBook Pro)/Desktop/Ryan_test_data/APS_2023Feb/test_sam3_s6_000655.ge3' 
    print(gefile)
    npzfile = '/Users/yetian/Dropbox/My Mac (Ye’s MacBook Pro)/Desktop/Ryan_test_data/APS_2023Feb/stainless_steel_test_75pct_bg_h_py_test.npz'
    print(npzfile)
    mypath = '/Users/yetian/Dropbox/My Mac (Ye’s MacBook Pro)/Desktop/Ryan_test_data/APS_2023Feb/' 
    print(mypath)
    num_images = 80 # number of frames in total in ge file
    omega = 0.25 # omega rotation angle in degree
    numframes = 80 # number of frames to export may be different with num_images which include some empty images
    bg_pct = 75 # background to subtract in percentile
    bg_nf = 30 # number of frames to use to generate dark field image               
    ge2npz(gefile, npzfile, mypath, num_images, omega, numframes, bg_pct, bg_nf)
