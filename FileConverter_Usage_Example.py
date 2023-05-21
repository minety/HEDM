import os
import sys

sys.path.append("/home/ytian37/scratch4-rhurley6/HEDM")  # add the directory to sys.path

from HEDMFileConverter import ToIlastikConverter, HDF5Slicer, run_conversion

def main():
    base_dir = '/Users/yetian/Desktop/Ryan_test_data/APS_2023Feb/nf_test/'
    input_file = base_dir + 'nugget1_nf_int_before/nugget1_nf_int4_000000.tif'
    slice_images = 1
    empty_images = 0
    num_images = 180 # number of frames in total in hdf5 file
    omega = 1 # omega rotation angle in degree
    bg_pct = 50 # background to subtract in percentile
    bg_nf = 180 # number of frames to use to generate dark field image   
    bgsub = True
    start_constant = 0 # replace with the constant you mentioned

    layers = 1 # replace with the actual value
    dets = 2 # replace with the actual value

    for layer in range(layers):
        for det in range(dets):
            output_file = base_dir + 'nugget1_nf_det{}_layer{}.h5'.format(det, layer)
            slice_output_file = base_dir + 'nugget1_nf_det{}_layer{}_50bg180nf_slice.h5'.format(det, layer)
            bgsub_h5_file = base_dir + 'nugget1_nf_det{}_layer{}_50bg.h5'.format(det, layer)
            
            start_num = start_constant + layer*num_images*omega*dets + det*num_images*omega
            end_num = start_num + num_images*omega -1

            print("Currently executing for layer: {}, det: {}, start_num: {}".format(layer, det, start_num))  # print current layer, det and start_num
            
            params = {
                'input_file': input_file,
                'output_file': output_file,
                'empty_images': empty_images, 
                'start_num': start_num,
                'end_num': end_num,
                'bgsub_h5_file': bgsub_h5_file,
                'num_images': num_images,
                'omega': omega,
                'bg_pct': bg_pct,
                'bg_nf': bg_nf,
                'bgsub': bgsub,
                'slice_input_file': bgsub_h5_file if bgsub else output_file,
                'slice_output_file': slice_output_file,
                'slice_images': slice_images
            }

            run_conversion(ToIlastikConverter, **params)
            run_conversion(HDF5Slicer, **params) 

if __name__ == "__main__":
    main()
