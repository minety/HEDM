import h5py
import argparse

def hdf5slice(input_hdf5, output_hdf5, numframes):
    # Open the original HDF5 file
    with h5py.File(input_hdf5, 'r') as f:
        # Get the image dataset from the original HDF5 file
        images = f['/imageseries/images']
        
        # Open the target HDF5 file
        with h5py.File(output_hdf5, 'w') as f_example:
            # Create the image dataset in the target HDF5 file and copy the first image from the original HDF5 file
            example_images = f_example.create_dataset('images', (numframes, images.shape[1], images.shape[2]), dtype=images.dtype)
            example_images[:] = images[:numframes,:,:]

if __name__ == '__main__':
    input_hdf5 = '/Users/yetian/Desktop/Ryan_test_data/APS_2023Feb/nf_test/nugget1_nf_int_det0_50bg.h5'
    output_hdf5 = '/Users/yetian/Desktop/Ryan_test_data/APS_2023Feb/nf_test/nugget1_nf_int_det0_50bg_slice.h5'
    numframes = 1
    hdf5slice(input_hdf5, output_hdf5, numframes)
