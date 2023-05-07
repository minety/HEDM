import os
import h5py
import re
from tifffile import imread

def create_hdf5(input_folder, output_file, start_num=None, end_num=None):
    image_paths = [os.path.join(input_folder, img_path) for img_path in os.listdir(input_folder) if img_path.lower().endswith('.tif') or img_path.lower().endswith('.tiff')]

    if len(image_paths) == 0:
        raise ValueError("No image paths found.")

    # Sort the image paths to ensure they are in the correct order
    image_paths = sorted(image_paths)

    def extract_number(img_path):
        return int(re.search(r'(\d{6})', os.path.splitext(os.path.basename(img_path))[0]).group())

    if start_num is not None and end_num is not None:
        image_paths = [img_path for img_path in image_paths if start_num <= extract_number(img_path) <= end_num]

    # Load the first image to get dimensions
    first_image = imread(image_paths[0])
    h, w = first_image.shape

    # Create an HDF5 file
    with h5py.File(output_file, 'w') as f:
        # Create a dataset with the shape (num_images, height, width) and the same dtype as the first image
        dataset = f.create_dataset('images', (len(image_paths), h, w), dtype=first_image.dtype)

        # Write the first image
        dataset[0] = first_image

        # Write the remaining images
        for i, img_path in enumerate(image_paths[1:], start=1):
            img = imread(img_path)
            dataset[i] = img

if __name__ == "__main__":
    input_folder = '/Users/yetian/Desktop/Ryan_test_data/APS_2023Feb/nf_test/nugget1_nf_int_before'
    output_file = '/Users/yetian/Desktop/Ryan_test_data/APS_2023Feb/nf_test/nugget1_nf_int_det1.hdf5'
    start_num = 0
    end_num = 179
    create_hdf5(input_folder, output_file, start_num, end_num)
