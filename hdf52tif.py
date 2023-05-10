import os
import h5py
from tifffile import TiffFile, imsave

def get_tiff_compression(tiff_path):
    with TiffFile(tiff_path) as tif:
        return tif.pages[0].compression

def hdf5_to_tiff(input_hdf5, output_folder, prefix, input_tiff_folder, start_num=None, end_num=None):
    with h5py.File(input_hdf5, 'r') as f:
        dataset = f['/exported_data']

        # The length of dataset
        num_images = len(dataset)

        # start and end idx
        start_idx = 0 if start_num is None else start_num
        end_idx = num_images if end_num is None else end_num + 1

        if start_idx >= num_images or end_idx > num_images:
            raise ValueError(f"Invalid range: start_num and end_num must be within the range of the dataset (0-{num_images - 1}).")

        # get the tiff compression method for nf-hedm at aps
        first_tiff_path = [os.path.join(input_tiff_folder, p) for p in os.listdir(input_tiff_folder) if p.lower().endswith('.tif') or p.lower().endswith('.tiff')][0]
        compression = get_tiff_compression(first_tiff_path)

        for i in range(start_idx, end_idx):
            img = dataset[i].squeeze()
            img[img < 500] = 0
            img_filename = f"{prefix}_{i + output_offset:06d}.tif"
            img_path = os.path.join(output_folder, img_filename)
            imsave(img_path, img, compression=compression)

if __name__ == "__main__":
    input_hdf5 = '/Users/yetian/Desktop/Ryan_test_data/APS_2023Feb/results_n3_det1.h5'
    output_folder = '/Users/yetian/Desktop/Ryan_test_data/APS_2023Feb/test_tiffs_headless_n3_det1'
    input_tiff_folder = '/Users/yetian/Desktop/Ryan_test_data/APS_2023Feb/nf_test/nugget1_nf_int_before'
    prefix = 'image'
    start_num = 0
    end_num = 2
    output_offset = 180
    os.makedirs(output_folder, exist_ok=True)
    hdf5_to_tiff(input_hdf5, output_folder, prefix, input_tiff_folder, start_num, end_num)
