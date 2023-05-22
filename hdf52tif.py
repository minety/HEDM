import os
import h5py
import re
from tifffile import TiffFile, imsave

def get_tiff_compression(tiff_path):
    with TiffFile(tiff_path) as tif:
        return tif.pages[0].compression
    
def sort_key(filename):
    match = re.match(r'nugget1_nf_layer(\d+)_det(\d+)_50bg_proc.h5', filename)
    if not match:
        return float('inf')  # 将不符合格式的文件排在最后
    layer, det = map(int, match.groups())
    return layer * 2 + det  # 按照 layer 和 det 的数值进行排序



def get_start_num_from_filename(filename):
    match = re.match(r'nugget1_nf_layer(\d+)_det(\d+)_50bg_proc.h5', filename)
    if not match:
        raise ValueError(f"Invalid filename: {filename}")
    layer, det = map(int, match.groups())
    return (layer * 2 + det) * 180

def hdf5_to_tiff(input_hdf5, output_folder, prefix, input_tiff_folder, start_num=None, end_num=None, output_offset=0):
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
            img[img < 50] = 0
            img_filename = f"{prefix}_{i + output_offset:06d}.tif"
            img_path = os.path.join(output_folder, img_filename)
            imsave(img_path, img, compression=compression)

if __name__ == "__main__":
    input_folder = '/Users/yetian/Desktop/Ryan_test_data/APS_2023Feb'
    output_folder = '/Users/yetian/Desktop/Ryan_test_data/APS_2023Feb/test_tiffs_headless_n3_det1'
    input_tiff_folder = '/Users/yetian/Desktop/Ryan_test_data/APS_2023Feb/nf_test/nugget1_nf_int_before'
    prefix = 'image'
    os.makedirs(output_folder, exist_ok=True)

    hdf5_files = sorted([f for f in os.listdir(input_folder) if f.lower().endswith('50bg_proc.h5')], key=sort_key)

    for hdf5_file in hdf5_files:
        input_hdf5 = os.path.join(input_folder, hdf5_file)
        output_offset = get_start_num_from_filename(hdf5_file)
        hdf5_to_tiff(input_hdf5, output_folder, prefix, input_tiff_folder, output_offset=output_offset)

