import numpy as np
from hexrd import imageseries
from hexrd.imageseries.omega import OmegaWedges
import sys, os
import h5py
import re
from tifffile import imread, TiffFile, imsave
import tifffile
import subprocess
import yaml
import argparse
import fabio
import time
import functools
import hdf5plugin
import numba
from xfab import tools
from ImageD11 import sparseframe, cImageD11, columnfile, grain, parameters
import pkg_resources
import matplotlib.pyplot as plt
from skimage import draw
from skimage import exposure
from matplotlib.patches import Circle
import textwrap
from scipy.sparse import coo_matrix
from tqdm import tqdm

class HEDM_Platform:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.base_dir = self.params['base_dir']
        self.sample_name = self.params['sample_name']
        self.input_file = self.params.get('input_file')
        self.imageseries_path = '/imageseries/images'
        self.image_default_path = '/images'
        self.image_input_path = self.params.get('image_input_path') 
        self.image_ilastik_path = '/exported_data'
        self.image_slice_path = None
        self.output_file = self.params.get('output_file')
        self.input_format = self._get_file_format()
        self.input_folder = self._get_input_folder()
        self.start_num = self.params.get('start_num', None)
        self.end_num = self.params.get('end_num', None)
        self.nf_threshold = self.params.get('nf_threshold', None) 
        self.num_images = self.params.get('num_images', None)
        self.omega = self.params.get('omega', None)
        self.bg_pct = self.params.get('bg_pct', None)
        self.nframes = self.params.get('nframes', None) 
        self.bg_nf = self.params.get('bg_nf', None) 
        self.bgsub_h5 = self.params.get('bgsub_h5_file', None)
        self.bgsub = self.params.get('bgsub', True)
        self.empty_images = self.params.get('empty_images', 0)
        self.slice_file = self.params.get('slice_file', False)
        self.slice_images = self.params.get('slice_images', 0)
        self.ilastik_loc = self.params.get('ilastik_loc')
        self.ilastik_project_file = self.params.get('ilastik_project_file', None)
        self.generate_hexomap_files = self.params.get('generate_hexomap_files', None)
        self.tiff_path = self.params.get('tiff_path', None)
        self.input_hdf5 = self.params.get('input_hdf5', None)
        self.output_offset = self.params.get('output_offset', None)
        self.hdf5_file_name = self.params.get('hdf5_file_name', None)
        self.hdf5_file = self.params.get('hdf5_file', None)
        self.generate_hexrd_files = self.params.get('generate_hexrd_files', True)
        self.generate_ImageD11_files = self.params.get('generate_ImageD11_files', True)
        self.flip_option = self.params.get('flip_option', None)
        self.ImageD11_process = self.params.get('ImageD11_process', False)
        self.NPKS = self.params.get('NPKS', 50)
        self.UPKS = self.params.get('UPKS', 45)
        self.Tolseq1 = self.params.get('Tolseq1', 0.03)
        self.Tolseq2 = self.params.get('Tolseq2', 0.025)
        self.Toldist = self.params.get('Toldist', 200)
        self.flt_THRESHOLDS = self.params.get('flt_THRESHOLDS', 75)
        self.Idx_prefix = self.params.get('Idx_prefix', 'test0')
        self.ilastik_proc = self.params.get('ilastik_proc', False)
        self.input_file_conv = self.params.get('input_file_conv', None)
        self.input_filepath = self.params.get('input_filepath', None)
        self.dataset_path = self.params.get('input_filepath', None)
        self.symmetry = self.params.get('symmetry')
        self.ring1 = self.params.get('ring1')
        self.ring2 = self.params.get('ring2')
        self.ilastik_input = self.params.get('ilastik_input', None)
        self.frame_number = self.params.get('frame_number', 0)
        self.proc_file = self.params.get('proc_file', None)
        self.raw_h5file = self.params.get('raw_h5file', None)
        self.output_png = self.params.get('output_png', None)
        self.tolangle = self.params.get('Tolangle', 1)
        self.r = self.params.get('Cylinder_radius', 800) 
        self.h = self.params.get('Search_height', 500)
        self.ncpu = self.params.get('Num_cpus', None)
        self.removalmask = np.ones((2048, 2048))
        self.center_position_x = self.params.get('center_position', {}).get('x', 1056)
        self.center_position_y = self.params.get('center_position', {}).get('y', 1005)
        self.max_num_frm = self.params.get('max_num_frm', 2)
        self.removal_regions = self.params.get('removal_regions', [])
        self.output_png_rm_reg = self.params.get('output_png_rm_reg', None)
        self.HEXRD_process = self.params.get('HEXRD_process', False)
        self.hexrd_findori_name = self.params.get('hexrd_findori_name', None)
        self.hexrd_fit_name = self.params.get('hexrd_fit_name', None) 
        self.multiprocessing = self.params.get('multiprocessing', -1)
        self.material_file = self.params.get('material_file', None)
        self.material_active = self.params.get('material_active', None)
        self.tth_width = self.params.get('tth_width', None)
        self.hexrd_npz_file = self.params.get('hexrd_npz_file', None)
        self.instrument_yml = self.params.get('instrument_yml', None)
        self.find_ori_threshold = self.params.get('find_ori_threshold', 2)
        self.find_ori_hkls = self.params.get('find_ori_hkls', None) 
        self.hkl_seeds = self.params.get('hkl_seeds', None)
        self.fiber_deg_seeds = self.params.get('fiber_deg_seeds', None)
        self.filter_radius_seeds = self.params.get('filter_radius_seeds', None)
        self.find_omega_tol = self.params.get('find_omega_tol', None)
        self.find_omega_period = self.params.get('find_omega_period', None)
        self.find_eta_tol = self.params.get('find_eta_tol', None)
        self.find_mis_ori_tol = self.params.get('find_mis_ori_tol', None)
        self.find_completeness = self.params.get('find_completeness', None)
        self.fit_threshold = self.params.get('fit_threshold', None)
        self.fit_tth_tol_seq = self.params.get('fit_tth_tol_seq', None)
        self.fit_eta_tol_seq = self.params.get('fit_eta_tol_seq', None)
        self.fit_omega_tol_seq = self.params.get('fit_omega_tol_seq', None) 
        self.refit_seq = self.params.get('refit_seq', None)
        self.tth_max = self.params.get('tth_max', None)
        self.input_hdf5_nf = self.params.get('input_hdf5_nf', None)
        self.output_hdf5_int_nf = self.params.get('output_hdf5_int_nf', None) 
        self.integration_factor = self.params.get('integration_factor', None) 

    def convert(self, *args, **kwargs):
        raise NotImplementedError("Subclass should implement this method")
    
    def _get_file_format(self):
        _, ext = os.path.splitext(self.input_file)
        return ext.lower()
    
    def _get_input_folder(self):
        return os.path.dirname(self.input_file)

    def tif2hdf5(self):
        input_file_prefix = os.path.splitext(os.path.basename(self.input_file))[0].rstrip("0123456789")
        image_paths = [os.path.join(self.input_folder, img_path) for img_path in os.listdir(self.input_folder) if (img_path.lower().endswith('.tif') or img_path.lower().endswith('.tiff')) and img_path.startswith(input_file_prefix)]

        self.start_num = int(self.start_num)
        self.end_num = int(self.end_num)

        if len(image_paths) == 0:
            print(f"No image paths found with the given prefix: {input_file_prefix}")
            return

        # Sort the image paths to ensure they are in the correct order
        image_paths = sorted(image_paths, key=lambda x: int(re.search(r'\d{6}', x).group()))

        if self.start_num is not None and self.end_num is not None:
            image_paths = image_paths[self.start_num:self.end_num+1]

        # Load the first image to get dimensions
        try:
            first_image = imread(image_paths[0])
        except Exception as e:
            print(f"Error loading the first image: {e}")
            return

        h, w = first_image.shape

        # Create an HDF5 file
        try:
            with h5py.File(self.output_file, 'w') as f:
                # Create a dataset with the shape (num_images, height, width) and the same dtype as the first image
                dataset = f.create_dataset('images', (len(image_paths), h, w), dtype=first_image.dtype)
                # Print the number of images in the dataset
                print(f"Number of images in the dataset: {len(image_paths)}")
                # Write the first image
                dataset[0] = first_image

                # Write the remaining images
                for i, img_path in enumerate(image_paths[1:], start=1):
                    try:
                        img = imread(img_path)
                    except Exception as e:
                        print(f"Error loading image {img_path}: {e}")
                        continue
                    dataset[i] = img
        except Exception as e:
            print(f"Error creating the HDF5 file: {e}")

    def make_meta():
        return {'testing': '1,2,3'}
    
    def get_file_info(self):
        if self.input_format in ('.hdf5', '.h5', '.nxs'):
            # Open the input HDF5 file
            with h5py.File(self.input_file, 'r') as h5_file:
                # Determine which path exists in the h5_file
                possible_paths = ['/imageseries/images', '/images', '/flyscan_00001/scan_data/orca_image']
                dataset_path = None
                for path in possible_paths:
                    if path in h5_file:
                        dataset_path = path
                        break

                # If no matching path is found, raise an error
                if not dataset_path:
                    raise ValueError(f"None of the expected paths {possible_paths} found in {self.input_file}")

                # Get the dataset from the determined path and print the info
                dataset = h5_file[dataset_path]
                print(f"File format: HDF5")
                print(f"File is located at path: {dataset_path}")
                print(f"Number of frames: {dataset.shape[0]}")

        elif self.input_format in ('.ge', '.ge2', '.ge3', '.ge5'):
            # Define the width and height of the images.
            width = 2048
            height = 2048

            # Read the raw data from the GE file.
            with open(self.input_file, 'rb') as f:
                # Skip the header part of the file.
                f.read(8192)
                # Read the rest of the file as a numpy array.
                data = np.fromfile(f, dtype=np.uint16)

            # Calculate the number of frames
            num_frames = len(data) // (width * height)
            
            print(f"File format: GE")
            print(f"Number of frames: {num_frames}")

        elif self.input_format == '.npz':
            # Open the npz file as an imageseries object
            ims = imageseries.open(
                self.input_file,
                format='frame-cache'
            )

            # Extract the omega vector and get its shape
            num_frames = np.array(ims.metadata['omega']).shape[0]

            # Print the file format and the number of frames
            print(f"File format: npz")
            print(f"Frames in the npz file: {num_frames}")

        else:
            print(f"Unsupported input format: {self.input_format}")

    def ge2hdf5(self):
        # Define the width and height of the images.
        # The default is 2048*2048 for most beamlines.
        width = 2048
        height = 2048

        # Read the raw data from the GE file.
        with open(self.input_file, 'rb') as f:
            # Skip the header part of the file.
            # 8192 bytes is the header size for APS sector 1.
            f.read(8192)

            # Read the rest of the file as a numpy array.
            # The data type is np.uint16.
            data = np.fromfile(f, dtype=np.uint16)

        # Convert the read data to an array of images.
        try:
            image_data = np.reshape(data, (self.num_images, height, width))
        except ValueError:
            raise ValueError("Please input the right total number of frames: num_images")

        # Remove the specified number of empty images.
        if self.empty_images > 0:
            image_data = image_data[self.empty_images:]

        # Create an HDF5 file and write the image data to the file.
        with h5py.File(self.output_file, 'w') as f:
            # Create a dataset in the HDF5 file.
            # The shape of the dataset is (len(image_data), height, width).
            # The data type is the same as the image_data array.
            dataset = f.create_dataset('images', (len(image_data), height, width), dtype=image_data.dtype)
            
            # Print the number of images in the dataset.
            print(f"Number of images in the dataset: {len(image_data)}")
            
            # Write the images to the dataset.
            for i, img in enumerate(image_data):
                dataset[i] = img

    def npz2hdf5(self):
        ims = imageseries.open(
            self.input_file,
            format='frame-cache'
        )

        # Input of omega metadata
        nf = self.nframes  # e.g., 720
        omega = self.omega
        omw = OmegaWedges(nf)
        omw.addwedge(0, nf*omega, nf) 
        ims.metadata['omega'] = omw.omegas

        # Encode all string metadata to UTF-8
        for key, value in list(ims.metadata.items()):
            if isinstance(value, str):
                ims.metadata[key] = value.encode('utf-8')
            elif isinstance(value, np.ndarray) and np.issubdtype(value.dtype, np.str_):
                ims.metadata[key] = np.char.encode(value, 'utf-8')

        # Now, apply the processing options
        ProcessedIS = imageseries.process.ProcessedImageSeries
        ops = [('flip', None)]
        pimgs = ProcessedIS(ims, ops)

        # Write to a temporary path
        temp_path = '/temp_imageseries'
        imageseries.write(pimgs, self.output_file, 'hdf5', path=temp_path)

        # Now, open the file and rename the group
        with h5py.File(self.output_file, 'a') as f:
            f.move(temp_path + '/images', '/images')
            del f[temp_path]

    def hdf5_to_hdf5(self):
        # Paths to check in the HDF5 file if dataset_path is not provided in the configuration
        possible_paths = ['/imageseries/images', '/images', '/flyscan_00001/scan_data/orca_image']

        # Open the input HDF5 file
        with h5py.File(self.input_file, 'r') as input_file:

            # If dataset_path is provided in the configuration, use it directly
            if self.dataset_path:
                if self.dataset_path not in input_file:
                    raise ValueError(f"Provided dataset_path {self.dataset_path} not found in {self.input_file}")
                dataset_path = self.dataset_path
            else:
                # Determine which path exists in the input_file
                dataset_path = None
                for path in possible_paths:
                    if path in input_file:
                        dataset_path = path
                        break
                # If no matching path is found, raise an error
                if not dataset_path:
                    raise ValueError(f"None of the expected paths {possible_paths} found in {self.input_file}")

            # Get the dataset from the determined path
            input_dataset = input_file[dataset_path]

            # Determine if there are empty images to be removed
            if self.empty_images > 0:
                if self.empty_images >= input_dataset.shape[0]:
                    raise ValueError(f"empty_images ({self.empty_images}) must be less than the number of frames ({input_dataset.shape[0]})")

                # Calculate the valid slice
                valid_slice = slice(self.empty_images, None)
            else:
                valid_slice = slice(None)

            # Open the output HDF5 file
            with h5py.File(self.output_file, 'w') as output_file:

                # If we are not skipping any frames, just create a hard link to the data
                if self.empty_images <= 0:
                    output_file['/images'] = h5py.ExternalLink(self.input_file, dataset_path)
                else:
                    # Use a virtual dataset to exclude empty frames and avoid copying data
                    num_frames = input_dataset.shape[0] - self.empty_images
                    output_shape = (num_frames,) + input_dataset.shape[1:]

                    # Create a VirtualSource for the input dataset
                    vsource = h5py.VirtualSource(input_file[dataset_path])

                    # Define a virtual layout for the new dataset
                    layout = h5py.VirtualLayout(shape=output_shape, dtype=input_dataset.dtype)

                    # Apply the slice to the virtual source
                    # Here we're assuming that 'valid_slice' applies to the 'frames' dimension
                    vsource_slice = (valid_slice, slice(None), slice(None))  # Applying valid_slice to the first dimension

                    # Map the valid portion of the input dataset onto the layout
                    layout[:] = vsource[vsource_slice]

                    # Create the virtual dataset in the output file under '/images'
                    output_dataset = output_file.create_virtual_dataset('/images', layout, fillvalue=0)

    def bgsub4hdf5(self, flip_option):
        ims = imageseries.open(
            self.output_file,
            format='hdf5',
            path='/',
            dataname='images'
        )

        # Input of omega meta data
        nf = self.num_images  #720
        omega = self.omega
        omw = OmegaWedges(nf)
        omw.addwedge(0, nf*omega, nf)
        ims.metadata['omega'] = omw.omegas

        # Make dark image from firstnf_to_use frames
        pct = self.bg_pct
        nf_to_use = self.bg_nf
        dark = imageseries.stats.percentile(ims, pct, nf_to_use)
        # np.save(DarkFile, dark)

        # Now, apply the processing options
        ProcessedIS = imageseries.process.ProcessedImageSeries
        ops = [('dark', dark), ('flip', flip_option)] # None, 'h', 'v', etc.
        pimgs = ProcessedIS(ims, ops)

        # Save the processed imageseries in hdf5 format
        imageseries.write(pimgs, self.bgsub_h5,'hdf5', path='/imageseries')
        return np.array(pimgs[0])

    # def validate_and_crop_images_to_new_file(self):
    #     possible_paths = ['/imageseries/images', '/images', '/flyscan_00001/scan_data/orca_image']
    #     original_file = self.bgsub_h5
    #     cropped_file = original_file.replace('.h5', '_cropped.h5')  # New file name with '_cropped'

    #     with h5py.File(original_file, 'r') as file:  # Open the original file in read mode
    #         # Try each path until we find the dataset
    #         for path in possible_paths:
    #             if path in file:
    #                 dataset = file[path]
    #                 first_image_shape = dataset[0].shape

    #                 # If the first image is not 2048x2048, create a new file and crop all images
    #                 if first_image_shape != (2048, 2048):
    #                     # Calculate the cropping coordinates
    #                     start_y = (first_image_shape[0] - 2048) // 2
    #                     start_x = (first_image_shape[1] - 2048) // 2
    #                     end_y = start_y + 2048
    #                     end_x = start_x + 2048

    #                     with h5py.File(cropped_file, 'w') as new_file:  # Create a new file in write mode
    #                         # Create a new dataset for cropped images with the same dataset name
    #                         cropped_shape = (len(dataset), 2048, 2048)
    #                         cropped_dataset = new_file.create_dataset(path, shape=cropped_shape, dtype=dataset.dtype)

    #                         # Iterate over all images to crop them and save to the new file
    #                         for i in range(len(dataset)):
    #                             # Crop the image
    #                             cropped_image = dataset[i, start_y:end_y, start_x:end_x]
    #                             # Write the cropped image to the new dataset
    #                             cropped_dataset[i, ...] = cropped_image

    #                         print(f"All images cropped to 2048x2048 and saved to new file {cropped_file}.")
    #                 else:
    #                     print(f"First image in {path} is already 2048x2048. No new file created.")
    #                 break
    #         else:
    #             print("None of the paths contained an image dataset.")

    def hdf5slice(self, input_file, slice_path):
        with h5py.File(input_file, 'r') as f:
            images = f[slice_path]
            base_name, extension = os.path.splitext(input_file)
            output_file = f"{base_name}_slice{extension}"
            with h5py.File(output_file, 'w') as f_example:
                example_images = f_example.create_dataset('images', (self.slice_images, images.shape[1], images.shape[2]), dtype=images.dtype)
                example_images[:] = images[:self.slice_images, :, :]
         
    # def sort_key(self):
    #     match = re.match(fr'{self.prefix}_layer(\d+)_det(\d+)_50bg_proc.h5', self.sort_filename)
    #     if not match:
    #         return float('inf')  # Place files that do not conform to the format at the end
    #     layer, det = map(int, match.groups())
    #     return layer * 2 + det  # Sort according to the numerical values of layer and det

    def get_start_num_from_filename(self, hdf5_file):
        sample_name = re.escape(self.sample_name)  # Escape to handle any special regex characters in the sample name
        # The pattern accounts for potential suffixes '_ilastik_proc' or '_proc' and '_50bg'
        pattern = fr"{sample_name}_layer(\d+)_det(\d+)_50bg(?:_ilastik_proc|_proc)?\.h5$"

        # Match the pattern against the actual HDF5 file name
        match = re.match(pattern, os.path.basename(hdf5_file))
        if not match:
            # If there is no match, raise an error indicating the filename is invalid
            raise ValueError(f"Invalid filename: {hdf5_file}")
        
        # Extract the layer and detector numbers and calculate the start number
        layer, det = map(int, match.groups())
        return (layer * 2 + det) * 180

    def integrate_images_in_hdf5(self, input_hdf5_path, output_hdf5_path, integration_factor=10):
        with h5py.File(input_hdf5_path, 'r') as input_file:
            input_dataset = input_file['images']
            num_images = input_dataset.shape[0]

            # Calculate the number of output images
            num_output_images = num_images // integration_factor
            if num_images % integration_factor:
                num_output_images += 1  # Account for the last incomplete batch
            
            # Open the output HDF5 file
            with h5py.File(output_hdf5_path, 'w', libver='latest') as output_file:
                # Create the output dataset
                output_dataset = output_file.create_dataset(
                    'images',
                    shape=(num_output_images,) + input_dataset.shape[1:],
                    dtype=np.int32,
                    chunks=True  # Enable chunking
                )
                
                # Initialize accumulation buffer
                accumulated_image = np.zeros(input_dataset.shape[1:], dtype=np.int32)
                count = 0
                output_index = 0
                
                # Process the images in chunks to reduce memory usage
                for i in tqdm(range(0, num_images, integration_factor), desc="Integrating images"):
                    # Determine the chunk size
                    chunk_end = min(i + integration_factor, num_images)
                    chunk = input_dataset[i:chunk_end].astype(np.int32)
                    
                    # Sum the images in the chunk
                    chunk_sum = np.sum(chunk, axis=0)
                    output_dataset[output_index] = chunk_sum
                    output_index += 1
                
                # Handle the last partial chunk if it exists
                if num_images % integration_factor:
                    # Only sum over the existing images in the last chunk
                    last_chunk = input_dataset[-(num_images % integration_factor):].astype(np.int32)
                    last_chunk_sum = np.sum(last_chunk, axis=0)
                    output_dataset[output_index] = last_chunk_sum

    def hdf5_to_tiff(self):
        base_name, extension = os.path.splitext(self.bgsub_h5)

        if self.ilastik_proc:
            # If ilastik processing is true, append '_ilastik_proc' to the base file name
            input_file_conv = f"{base_name}_ilastik_proc{extension}"
        else:
            # If ilastik_proc is false, use the user-provided input_file_conv if available
            # Otherwise, default to appending '_proc' to the base file name
            input_file_conv = self.input_file_conv if self.input_file_conv is not None else f"{base_name}{extension}"

        print(f"Input file for conversion: {input_file_conv}")  # Print the input_file_conv file name

        # Paths to check in the HDF5 file if dataset_path is not provided in the configuration
        possible_paths = ['/imageseries/images', '/images', '/flyscan_00001/scan_data/orca_image']

        # Open the input HDF5 file
        with h5py.File(input_file_conv, 'r') as input_file:

            # If dataset_path is provided in the configuration, use it directly
            if self.dataset_path:
                if self.dataset_path not in input_file:
                    raise ValueError(f"Provided dataset_path {self.dataset_path} not found in {input_file_conv}")
                dataset_path = self.dataset_path
            else:
                # Determine which path exists in the input_file
                dataset_path = None
                for path in possible_paths:
                    if path in input_file:
                        dataset_path = path
                        break
                # If no matching path is found, raise an error
                if not dataset_path:
                    raise ValueError(f"None of the expected paths {possible_paths} found in {input_file_conv}")

            # Get the dataset from the determined path
            input_dataset = input_file[dataset_path]
            output_offset = self.get_start_num_from_filename(input_file_conv)  # Removed the input_file_conv as it's not used in the method
            print(f"Output offset: {output_offset}")
            # Currently using 180 degrees
            num_images = 180
            print(f"Number of images: {num_images}") 
            # start and end idx
            start_idx = 0
            end_idx = num_images
            if start_idx >= num_images or end_idx > num_images:
                raise ValueError(f"Invalid range: start_num and end_num must be within the range of the dataset (0-{num_images - 1}).")
            # Set the TIFF compression method as a constant
            TIFF_COMPRESSION = None
            compression = TIFF_COMPRESSION
            start_idx = int(start_idx)
            end_idx = int(end_idx)

            # Construct the output folder name using sample_name and nf
            tiff_output_folder = os.path.join(self.base_dir, f"{self.sample_name}_nf")

            # Ensure the output directory exists, create if it does not
            os.makedirs(tiff_output_folder, exist_ok=True)

            for i in range(start_idx, end_idx):
                img = input_dataset[i].squeeze()
                # Check if the image needs to be cropped to 2048x2048
                if img.shape != (2048, 2048):
                    # Calculate the cropping coordinates (assuming you want to crop the center of the image)
                    start_y = (img.shape[0] - 2048) // 2
                    start_x = (img.shape[1] - 2048) // 2
                    end_y = start_y + 2048
                    end_x = start_x + 2048
                    # Crop the image
                    img = img[start_y:end_y, start_x:end_x]

                img[img < self.nf_threshold] = 0
                # Convert the image to float32 before saving
                img = img.astype(np.float32)
                img_filename = f"{self.sample_name}_nf_{i + output_offset:06d}.tif"
                img_path = os.path.join(tiff_output_folder, img_filename)
                imsave(img_path, img, compression=compression)

    # def get_layer_from_filename(self):
    #     match = re.match(fr'{self.prefix}_layer(\d+)_det(\d+)_50bg_proc.h5', self.sort_filename)
    #     if not match:
    #         raise ValueError(f"Invalid filename: {self.sort_filename}")
    #     layer, det = map(int, match.groups())
    #     return layer

    def hdf5_to_npz(self):
        base_name, extension = os.path.splitext(self.bgsub_h5)

        if self.ilastik_proc:
            # If ilastik processing is true, append '_ilastik_proc' to the base file name
            input_file_conv = f"{base_name}_ilastik_proc{extension}"
            output_file = f"{base_name}_ilastik_proc.npz" 
        else:
            # If ilastik_proc is false, use the user-provided input_file_conv if available
            # Otherwise, default to appending '_proc' to the base file name
            input_file_conv = self.input_file_conv if self.input_file_conv is not None else f"{base_name}{extension}"
            output_file = f"{base_name}_proc.npz"

        # Check for the possible data paths in the HDF5 file
        with h5py.File(input_file_conv, 'r') as hf:
            if 'images' in hf:
                dataname = 'images'
            elif 'imageseries/images' in hf:
                dataname = 'imageseries/images'
            else:
                raise ValueError("The HDF5 file does not contain either 'images' or 'imageseries/images' data paths.")

        ims = imageseries.open(
            input_file_conv,
            format='hdf5',
            path='/',
            dataname=dataname
        )

        # Input of omega meta data
        nf = self.nframes  #720
        omega = self.omega
        omw = OmegaWedges(nf)
        omw.addwedge(0, nf*omega, nf) 
        ims.metadata['omega'] = omw.omegas

        # Now, apply the processing options
        ProcessedIS = imageseries.process.ProcessedImageSeries
        # ops = [('dark', dark), ('flip', None)]  # Comment out to reduce computing time
        ops = [('flip', None)]
        pimgs = ProcessedIS(ims, ops)

        # Save the processed imageseries in npz format
        print(f"Writing npz file (may take a while): {output_file}")
        imageseries.write(pimgs, output_file, 'frame-cache', threshold=5, cache_file=output_file)

    ########################### 
    # ImageD11 related functions
    ###########################   

    def hdf5_to_bg_edf(self):
        base_name, extension = os.path.splitext(self.bgsub_h5)

        if self.ilastik_proc:
            # If ilastik processing is true, append '_ilastik_proc' to the base file name
            input_file_conv = f"{base_name}_ilastik_proc{extension}"
        else:
            # If ilastik_proc is false, use the user-provided input_file_conv if available
            # Otherwise, default to appending '_proc' to the base file name
            input_file_conv = self.input_file_conv if self.input_file_conv is not None else f"{base_name}{extension}"
        
        # Open the ilastik processed file
        with h5py.File(input_file_conv, 'r') as f:
            # Check for the possible data paths in the HDF5 file
            if 'images' in f:
                dataname = 'images'
            elif 'imageseries/images' in f:
                dataname = 'imageseries/images'
            else:
                raise ValueError("The HDF5 file does not contain either 'images' or 'imageseries/images' data paths.")
            
            im = f[dataname]
            im2 = np.median(im[:, :, :], axis=0)  # Calculate median for all frames
            
        # Save background
        output_filename = f"{base_name}_bg.edf"
        fabio.edfimage.edfimage(im2.astype(np.uint16)).write(output_filename)

    def run_sparse_script(self):
        script_path = pkg_resources.resource_filename('HEDM_Platform', 'scripts/sparse_h5.py')
        
        # Construct the input and output filenames
        base_name, extension = os.path.splitext(self.bgsub_h5)
        
        if self.ilastik_proc:
            # If ilastik processing is true, append '_ilastik_proc' to the base file name
            input_file_conv = f"{base_name}_ilastik_proc{extension}"
        else:
            # If ilastik_proc is false, use the user-provided input_file_conv if available
            # Otherwise, default to appending '_proc' to the base file name
            input_file_conv = self.input_file_conv if self.input_file_conv is not None else f"{base_name}{extension}"
        
        bg_filename = f"{base_name}_bg.edf"
        
        # Construct the full command
        cmd = [
            'python', script_path,
            input_file_conv,
            bg_filename,
            str(self.flt_THRESHOLDS),
            str(self.num_images),
            str(self.base_dir)
        ]
        
        # Execute the command
        subprocess.run(cmd)

    def run_gen_flt_script(self):
        script_path = pkg_resources.resource_filename('HEDM_Platform', 'scripts/gen_flt.py')
        
        # Construct the input and output filenames
        base_name, extension = os.path.splitext(self.bgsub_h5)

        if self.ilastik_proc:
            # If ilastik processing is true, append '_ilastik_proc' to the base file name
            input_file_conv = f"{base_name}_ilastik_proc{extension}"
            sparse_file = f"{base_name}_ilastik_proc_t{str(self.flt_THRESHOLDS)}_sparse.h5"
            flt_file = f"{base_name}_ilastik_proc_t{str(self.flt_THRESHOLDS)}.flt"

        else:
            # If ilastik_proc is false, use the user-provided input_file_conv if available
            # Otherwise, default to appending '_proc' to the base file name
            input_file_conv = self.input_file_conv if self.input_file_conv is not None else f"{base_name}{extension}"

            # Remove 'ilastik_' when input_file_conv is provided
            sparse_file = f"{base_name}_t{str(self.flt_THRESHOLDS)}_sparse.h5"
            flt_file = f"{base_name}_t{str(self.flt_THRESHOLDS)}.flt"

        par_file = base_name.replace(f"_{self.bg_pct}bg", "") + ".par"
 
        # Construct the full command
        cmd = [
            'python', script_path, 
            input_file_conv,
            sparse_file,
            flt_file,
            par_file,
        ]
        # Execute the command
        subprocess.run(cmd)

    def run_clean_flt_script(self):
        script_path = pkg_resources.resource_filename('HEDM_Platform', 'scripts/clean_flt.py')

        # Construct the input and output filenames
        base_name, extension = os.path.splitext(self.bgsub_h5) 
        
        # Additional filenames based on your requirements
        merge3D_tor = 1.6
        par_file = base_name.replace(f"_{self.bg_pct}bg", "") + ".par"
        base_dir = os.path.dirname(base_name)
        # Determine the flt_file_name based on the presence of input_file_ImageD11

        if self.ilastik_proc:
            # If ilastik processing is true, append '_ilastik_proc' to the base file name
            flt_file_name = os.path.basename(f"{base_name}_ilastik_proc_t{str(self.flt_THRESHOLDS)}.flt")
        else:
            # If ilastik_proc is false, use the user-provided input_file_conv if available
            # Otherwise, default to appending '_proc' to the base file name
            flt_file_name = os.path.basename(f"{base_name}_t{str(self.flt_THRESHOLDS)}.flt")       

        # Construct the full command
        cmd = [
            'python', script_path, 
            base_dir,
            flt_file_name,
            par_file,
            flt_file_name,  # Clean and overwrite .flt file
            str(merge3D_tor) # 1.6 as default, no need to modify
        ]
        
        # Execute the command
        subprocess.run(cmd)

    def run_ImageD11_indexing_script(self):
        script_path = pkg_resources.resource_filename('HEDM_Platform', 'scripts/indexing.py')

        # Construct the input and output filenames
        base_name, extension = os.path.splitext(self.bgsub_h5) 

        # Additional filenames based on your requirements
        par_file = base_name.replace(f"_{self.bg_pct}bg", "") + ".par"
        Idx_file_prefix = self.Idx_prefix
        
        if self.ilastik_proc:
            # If ilastik processing is true, append '_ilastik_proc' to the base file name
            flt_file_name = os.path.basename(f"{base_name}_ilastik_proc_t{str(self.flt_THRESHOLDS)}.flt")
        else:
            # If ilastik_proc is false, use the user-provided input_file_conv if available
            # Otherwise, default to appending '_proc' to the base file name
            flt_file_name = os.path.basename(f"{base_name}_t{str(self.flt_THRESHOLDS)}.flt")  

        NPKS = self.NPKS
        UPKS = self.UPKS
        Tolseq1 = self.Tolseq1
        Tolseq2 = self.Tolseq2
        Toldist = self.Toldist

        symmetry = self.symmetry
        ring1 = ",".join(map(str, self.ring1))
        ring2 = ",".join(map(str, self.ring2))

        # Extract the new parameters
        tolangle = str(self.tolangle)  # Assuming tolangle is a float or int
        r = str(self.r)                # Assuming r is an int
        h = str(self.h)              # Assuming r2 is an int
        ncpu = str(self.ncpu)

        # Set the OMP_NUM_THREADS environment variable
        os.environ['OMP_NUM_THREADS'] = '1'
        
        # Construct the full command
        cmd = [
            'python', script_path, 
            flt_file_name,
            par_file,
            Idx_file_prefix,
            str(NPKS),
            str(UPKS),
            str(Tolseq1),
            str(Tolseq2),
            str(Toldist),
            symmetry,
            ring1,
            ring2,
            tolangle,  
            r,         
            h,
            ncpu    
        ]
        
        print("Executing command:", cmd)
        # Execute the command
        subprocess.run(cmd)


    def run_ImageD11_fitting_script(self, u_file, U_file, par_file, flt_file_name, t, omega_slop):
        script_path = pkg_resources.resource_filename('HEDM_Platform', 'scripts/fitting.py')
        cmd = [
            'python', script_path,
            '-u', u_file,
            '-U', U_file,
            '-p', par_file,
            '-f', flt_file_name,
            '-t', str(t),
            '--omega_slop', str(omega_slop)
        ]
        subprocess.run(cmd)

    def run_ImageD11_all_fitting_scripts(self):
        base_name, extension = os.path.splitext(self.bgsub_h5)
        par_file = base_name.replace(f"_{self.bg_pct}bg", "") + ".par"

        if self.ilastik_proc:
            # If ilastik processing is true, append '_ilastik_proc' to the base file name
            flt_file_name = os.path.basename(f"{base_name}_ilastik_proc_t{str(self.flt_THRESHOLDS)}.flt")
        else:
            # If ilastik_proc is false, use the user-provided input_file_conv if available
            # Otherwise, default to appending '_proc' to the base file name
            flt_file_name = os.path.basename(f"{base_name}_t{str(self.flt_THRESHOLDS)}.flt")  

        t_values = [0.05, 0.04, 0.03, 0.02, 0.01]
        omega_slop = 0.05
        
        u_file = f'all{self.Idx_prefix}.map'
        for i, t in enumerate(t_values):
            U_file = f'all{self.Idx_prefix}_it{i+1}.map'
            self.run_ImageD11_fitting_script(u_file, U_file, par_file, flt_file_name, t, omega_slop)
            u_file = U_file

        # Calculate stress tensors
        self.run_stress_calculation(U_file, f"{base_name}_pos_stress.txt", par_file)

    def run_stress_calculation(self, inputmap, outfile, useParFile):
        ## Open par file
        pars = parameters.read_par_file(useParFile)
        unit_cell = [pars.get('cell__a'), pars.get('cell__b'), pars.get('cell__c'), pars.get('cell_alpha'), pars.get('cell_beta'), pars.get('cell_gamma')]

        ## Open latest map
        gl = grain.read_grain_file(inputmap)

        ## Compute strains and stress in sample frame
        strain_tensors_sample = []
        stress_tensors_sample = []
        rods_sample = []
        vol_strains = []
        stiffness2 = np.array( [[87.26e9, 6.57e9, 11.95e9, -17.18e9, 0, 0], 
                                [6.57e9, 87.26e9, 11.95e9, 17.18e9, 0, 0], 
                                [11.95e9, 11.95e9, 105.8e9, 0, 0, 0], 
                                [-17.18e9, 17.18e9, 0, 57.15e9, 0, 0], 
                                [0, 0, 0, 0, 57.15e9, 0], 
                                [0, 0, 0, 0, -17.18e9, 0.5*(87.26e9-6.57e9)]])
        for g in gl:
            u, strain_crystal = tools.ubi_to_u_and_eps(g.ubi, unit_cell)
            tmprod = g.translation
            e11, e12, e13, e22, e23, e33 = strain_crystal
            strain_crystal_voigt = np.transpose(np.array([e11, e22, e33, e23, e13, e12]))
            stress_crystal_voigt = stiffness2.dot(strain_crystal_voigt)
            s11, s22, s33, s23, s13, s12 = stress_crystal_voigt
            strain_tensor_crystal = np.array([[e11, e12, e13], [e12, e22, e23], [e13, e23, e33]])
            stress_tensor_crystal = np.array([[s11, s12, s13], [s12, s22, s23], [s13, s23, s33]])
            strain_tensor_sample = u.dot(strain_tensor_crystal.dot(u.T))
            stress_tensor_sample = u.dot(stress_tensor_crystal.dot(u.T))
            vol_strain = (strain_tensor_sample[0, 0] + strain_tensor_sample[1, 1] + strain_tensor_sample[2, 2]) / 3.
            vol_strains.append(vol_strain)
            rods_sample.append(tmprod)
            strain_tensors_sample.append(strain_tensor_sample)
            stress_tensors_sample.append(stress_tensor_sample)

        vol_strains = np.array(vol_strains)
        max_vol_strains = np.max(vol_strains)
        print('max_vol_strains', max_vol_strains)

        ## Print
        with open(outfile, 'w') as f:
            header_items = ('xpos (um) ypos (um) zpos (um) s11 (Pa) s22 (Pa) s33 (Pa) s23 (Pa) s13 (Pa) s13 (Pa)')
            f.write(header_items + "\n")
            for rod, stress_tensor in zip(rods_sample, stress_tensors_sample):
                f.write(f"{rod[0]}, {rod[1]}, {rod[2]}, {stress_tensor[0, 0]}, {stress_tensor[1, 1]}, {stress_tensor[2, 2]}, {stress_tensor[1, 2]}, {stress_tensor[0, 2]}, {stress_tensor[0, 1]}\n")

    ###########################
    # HEXRD related functions
    ###########################

    def create_hexrd_config(self, analysis_name):
        # Create the YAML configuration as a formatted string
        config = textwrap.dedent(f"""
        analysis_name: {analysis_name}

        working_dir: {self.base_dir}
        multiprocessing: {self.multiprocessing} # "all", or "half", or -1 means all but one, defaults to -1

        material:
          definitions: {self.material_file}
          active: {self.material_active}
          dmin: 1 # defaults to 1.0 angstrom
          tth_width: {self.tth_width} # defaults to 0.25 degrees
          # min_sfac_ratio: 0.05 # min percentage of max |F|^2 to exclude; default None

        image_series:
          format: frame-cache
          data:
            - file: {self.hexrd_npz_file}
              args: {{}}
              panel: GE  # must match detector key

        instrument: {self.instrument_yml}

        find_orientations:
          orientation_maps:
            file: null
            threshold: {self.find_ori_threshold}  # find_orientations
            bin_frames: 1 # defaults to 1
            active_hkls: {self.find_ori_hkls} # opthkls

          seed_search: # this section is ignored if use_quaternion_grid is defined
            hkl_seeds: {self.hkl_seeds} # hkls ids to use, must be defined for seeded search
            fiber_step: {self.fiber_deg_seeds}  # degrees, defaults to ome tolerance

            method:
              label:
                filter_radius: {self.filter_radius_seeds}
                threshold: 1 # defaults to 1

          threshold: 1 # scoring?
          omega:
            tolerance: {self.find_omega_tol}  # in degrees, defaults to 2x ome step
            period: {self.find_omega_period}   #[180, -180]

          eta:
            tolerance: {self.find_eta_tol}  # in degrees, defaults to 2x ome step
            mask: 5  # degrees, mask angles close to ome rotation axis, defaults to 5

          clustering:
            radius: {self.find_mis_ori_tol}   # misorientation search range
            completeness: {self.find_completeness} # completeness threshold
            algorithm: dbscan  

        analysis_name: {analysis_name}

        fit_grains:
          do_fit: true # if false, extracts grains but doesn't fit. defaults to true
          estimate: {self.hexrd_findori_name}/grains.out
          npdiv: 2 # number of polar pixel grid subdivisions, defaults to 2
          threshold: {self.fit_threshold}
          tolerance:
            tth: {self.fit_tth_tol_seq}    #[0.25, 0.20] # tolerance lists must be identical length
            eta: {self.fit_eta_tol_seq}    #[2.0, 1.0]
            omega: {self.fit_omega_tol_seq}  #[2.0, 1.0]
          refit: {self.refit_seq}  #[2, 1.1]
          tth_max: {self.tth_max}  #9.344  # true, false, or a non-negative value, defaults to true
        """) 

        hexrd_config_file = f'{analysis_name}_hexrd.yml'
        # Save the configuration file
        with open(hexrd_config_file, 'w') as f:
            f.write(config)

        print(f"Config file saved as {hexrd_config_file}")

    def run_hexrd_find_orientations(self, analysis_name):
        hexrd_config_file = f'{analysis_name}_hexrd.yml'
        # Run 'hexrd find-orientations'
        cmd = ['hexrd', 'find-orientations', hexrd_config_file]
        subprocess.run(cmd)

    def run_hexrd_fit_grains(self, analysis_name):
        hexrd_config_file = f'{analysis_name}_hexrd.yml'
        # Run 'hexrd fit-grains'
        cmd = ['hexrd', 'fit-grains', hexrd_config_file]
        subprocess.run(cmd)

    def standard_hdf5(self, input_filepath, input_dataset_name, output_dataset_name):
        # Temporary output file path
        output_filepath = input_filepath + '_temp'
        # Open the original file
        with h5py.File(input_filepath, 'r') as f:
            data = f[input_dataset_name][:]  # load the data into memory
        # Remove the last dimension
        data_squeezed = np.squeeze(data, axis=-1)
        # Save the squeezed data to a new HDF5 file
        with h5py.File(output_filepath, 'w') as f:
            f.create_dataset(output_dataset_name, data=data_squeezed)
        # Delete the original file
        os.remove(input_filepath)
        # Rename the temporary output file to the original file name
        os.rename(output_filepath, input_filepath)

    def load_frame(self, filename):
        with h5py.File(filename, 'r') as f:
            if 'images' in f:
                group = f['images']
                if isinstance(group, h5py.Dataset):
                    return group[self.frame_number]
                else:
                    for name, ds in group.items():
                        if isinstance(ds, h5py.Dataset):
                            return ds[self.frame_number]
            elif 'imageseries/images' in f:
                group = f['imageseries/images']
                if isinstance(group, h5py.Dataset):
                    return group[self.frame_number]
                else:
                    for name, ds in group.items():
                        if isinstance(ds, h5py.Dataset):
                            return ds[self.frame_number]
            else:
                raise ValueError("Neither 'images' nor 'imageseries/images' were found in the HDF5 file.")

    def load_peaks(self, filename):
        file_extension = filename.split('.')[-1]
        if file_extension == "flt":
            colf = columnfile.columnfile(filename)
            peaks_s = colf.sc
            peaks_f = colf.fc
            peaks_omega = colf.omega
        elif file_extension == "npz":
            # Placeholder for the npz loading logic
            peaks_s = np.array([])
            peaks_f = np.array([])
            peaks_omega = np.array([])
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")

        omega_step = 360 / self.num_images  # Calculate omega step based on number of images

        # Calculate the specific omega value for the given frame_number
        desired_omega = self.frame_number * omega_step

        # Create a mask for peaks that are close to the desired omega (considering minor floating-point discrepancies)
        mask = np.isclose(peaks_omega, desired_omega, atol=1e-5)

        # Printing the number of spots found
        num_spots = np.sum(mask)
        print(f"Found {num_spots} spots for frame {self.frame_number} (omega: {desired_omega:.2f} degrees).")

        return peaks_s[mask], peaks_f[mask]

    def visualize_difference(self):
        raw_frame = self.load_frame(self.raw_h5file)

        # Create a figure and axis
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Adjust contrast based on the min and max values for better visibility
        vmin = np.min(raw_frame)
        vmax = np.max(raw_frame)
        ax.imshow(raw_frame, cmap='gray_r', origin='lower', vmin=vmin, vmax=vmax)

        # Check if it's an h5 file and plot its content
        if self.proc_file.endswith('.h5'):
            proc_frame = self.load_frame(self.proc_file)
            mask = proc_frame != 0
            ax.imshow(mask, cmap='Reds', origin='lower', alpha=0.6)

        # Check if it's an flt file and plot its content
        elif self.proc_file.endswith('.flt'):
            proc_s, proc_f = self.load_peaks(self.proc_file)

            print("Coordinates of first few spots:")
            for s, f in zip(proc_s[:5], proc_f[:5]):
                print(f"Spot: s={s:.4f}, f={f:.4f}")

            # Overlay empty circles on the raw frame
            for s, f in zip(proc_s, proc_f):
                if 0 <= s < raw_frame.shape[0] and 0 <= f < raw_frame.shape[1]:
                    circle = Circle((f, s), radius=15, edgecolor='r', facecolor='none', linewidth=1.5, alpha=0.5)
                    ax.add_patch(circle)

        # Placeholder for the npz loading logic
        elif self.proc_file.endswith('.npz'):
            pass

        else:
            raise ValueError(f"Unsupported file extension for processed file: {self.proc_file.split('.')[-1]}")

        plt.savefig(self.output_png)

    def update_removal_mask(self):
        cx, cy = self.center_position_x, self.center_position_y
        
        # Create arrays of x and y indices
        x_indices, y_indices = np.meshgrid(np.arange(2048), np.arange(2048), indexing='ij')
        
        # Compute the distance array from the center
        distances = np.sqrt((x_indices - cx)**2 + (y_indices - cy)**2)
        
        for region in self.removal_regions:
            start, end = region["start"], region["end"]
            should_apply = region.get("apply", True)

            if should_apply:
                mask_region = (distances >= start) & (distances < end)
                self.removalmask[mask_region] = 0
        
        # Construct absolute file path
        file_path = os.path.join(self.base_dir, 'removalmask.npy')
        
        # Save removalmask using the absolute path
        np.save(file_path, self.removalmask) 

    def visualize_removal_regions(self):
        # Open the input HDF5 file
        with h5py.File(self.bgsub_h5, 'r') as h5_file:
            # Determine which path exists in the h5_file
            possible_paths = ['/imageseries/images', '/images', '/flyscan_00001/scan_data/orca_image']
            dataset_path = None
            for path in possible_paths:
                if path in h5_file:
                    dataset_path = path
                    break

            # If no matching path is found, raise an error
            if not dataset_path:
                raise ValueError(f"None of the expected paths {possible_paths} found in {self.bgsub_h5}")

            # Get the dataset from the determined path
            dataset = h5_file[dataset_path]

            # Extract up to max_num_frm frames and compute their maximum
            max_frame = np.max(dataset[:self.max_num_frm], axis=0)

        # Reverse the colors: higher values become black
        reversed_frame = np.max(max_frame) - max_frame

        # Plot the reversed frame
        plt.imshow(reversed_frame, cmap='gray')

        # Overlay the removal mask regions in red with 30% transparency
        plt.imshow(np.ma.masked_where(self.removalmask == 1, self.removalmask), cmap='Reds', alpha=0.8)  

        # Save the plot as PNG
        plt.savefig(self.output_png_rm_reg, dpi=300)
        plt.close()

class Check_file_info(HEDM_Platform):
    def convert(self):
        self.get_file_info()

class Removal_Mask(HEDM_Platform):
    def convert(self):
        self.update_removal_mask()
        self.visualize_removal_regions()

class Standardize_format(HEDM_Platform):
    def convert(self):
        print("Standardizing format...")
        if self.input_format in ('.ge', '.ge2', '.ge3', '.ge5'):
            print("Converting from .ge format to .h5...")
            self.ge2hdf5()
        elif self.input_format in ('.tif', '.tiff'):
            print("Converting from .tif format to .h5...")
            self.tif2hdf5()
        elif self.input_format in ('.hdf5', '.h5', '.nxs'):  
            print("Converting from .h5/.nxs format to .h5...")
            self.hdf5_to_hdf5()
        elif self.input_format in ('.npz'):  
            print("Converting from .npz format to .h5...")
            self.npz2hdf5()
        else:
            print(f"Unsupported input format: {self.input_format}")

class Integrate_images_in_hdf5(HEDM_Platform):
    def convert(self):
        self.integrate_images_in_hdf5(self.input_hdf5_nf, self.output_hdf5_int_nf, self.integration_factor)

class Subtract_background(HEDM_Platform):
    def convert(self):
        if self.bgsub:
            print("Applying background subtraction...")
            self.bgsub4hdf5(self.flip_option)
            print(f"HEXRD flip option is '{self.flip_option}'.")
            # self.validate_and_crop_images_to_new_file()

class Process_with_ilastik(HEDM_Platform):
    def convert(self):
        print(f'ilastik_project_file is {self.ilastik_project_file}')
        if self.ilastik_proc:
            print("Processing with ilastik...")
            base_name, extension = os.path.splitext(self.bgsub_h5) 
            output_file = f"{base_name}_ilastik_proc{extension}"
            
            # Choose the correct input based on the condition
            ilastik_input = self.ilastik_input if self.ilastik_input is not None else self.bgsub_h5
            
            # Run the non-parallel code
            subprocess.run([
                self.ilastik_loc,
                '--headless',
                f'--cutout_subregion=[(0,0,0,0),({self.nframes},2048,2048,1)]',
                '--pipeline_result_drange=(0.0,1.0)',
                '--export_drange=(0,100)',
                f'--output_filename_format={output_file}',
                f'--project={self.ilastik_project_file}',
                '--export_source=Probabilities',
                '--raw_data='+f'{ilastik_input}'  # Use the chosen input
            ], env={'LAZYFLOW_THREADS': '6'})
            
            self.standard_hdf5(output_file, self.image_ilastik_path, self.image_default_path)
    
        if self.slice_file:
            print("Generating sliced ilastik processed data...")
            self.hdf5slice(output_file, self.image_default_path)

class Convert_to_hedm_formats(HEDM_Platform):
    def convert(self):
        if self.generate_hexomap_files:
            print("Generating hexomap .tif files...")
            self.hdf5_to_tiff()

        if self.generate_hexrd_files:
            print("Generating hexrd .npz files...")
            self.hdf5_to_npz()

        if self.generate_ImageD11_files:
            print("Preparing ImageD11 .edf bg files...")
            self.hdf5_to_bg_edf()
            print("Generating ImageD11 .h5 sparse files...")
            self.update_removal_mask()
            self.run_sparse_script()
            print("Generating ImageD11 .flt files...")
            self.run_gen_flt_script()
            print("Cleaning ImageD11 .flt files...")
            self.run_clean_flt_script()

        if not any([self.generate_hexomap_files, self.generate_hexrd_files, self.generate_ImageD11_files]):
            print("None of the generate flags is set to True. No action will be performed.")

class SliceHDF5(HEDM_Platform):
    def convert(self):
        print("Slicing HDF5 file...")
        self.hdf5slice(self.params.get('slice_input_file'), self.params.get('input_file_path'))

class ff_HEDM_process(HEDM_Platform):
    def convert(self):
        if self.ImageD11_process:
            print("ImageD11 indexing...")
            self.run_ImageD11_indexing_script()
            print("ImageD11 fitting...")
            self.run_ImageD11_all_fitting_scripts()

        if self.HEXRD_process:
            print("Generating .yml file for Hexrd...")
            print("Hexrd finding orientation...")
            self.create_hexrd_config(self.hexrd_findori_name)
            self.run_hexrd_find_orientations(self.hexrd_findori_name)
            print("Hexrd fitting grains...")
            self.create_hexrd_config(self.hexrd_fit_name)
            self.run_hexrd_fit_grains(self.hexrd_fit_name)
        
        if not (self.ImageD11_process or self.HEXRD_process):
            print("No processing option selected. Please select at least one processing option.")

class Visualize_Diff(HEDM_Platform):
    def convert(self):
        print("Saving the overlapping files...")
        self.visualize_difference()

def run_function(converter_class, **params):
    converter = converter_class(**params)
    converter.convert()

def load_config(file_path):
    with open(file_path, 'r') as stream:
        try:
            return yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
            return None


def main():
    parser = argparse.ArgumentParser(description='Process some parameters.')
    parser.add_argument('command', choices=['check_file','rm_mask', 'int_images', 'stand', 'sub', 'ilastik', 'hedm_formats', 'all', 'slice', 'ff_HEDM_process', 'vis_diff'])
    parser.add_argument('config_file')
    args = parser.parse_args()

    if len(sys.argv) != 3:
        print('Usage: HEDM_Platform <command> <config_file>')
        sys.exit()

    # Load configuration
    config_file = args.config_file
    params = load_config(config_file)
    if params is None:
        print('Error loading configuration')
        sys.exit()

    # Calculate nframes and add it to the params
    params['nframes'] = params['num_images'] - params['empty_images']

    # # Create output directory only if generate_hexomap_files is True
    # if params.get('generate_hexomap_files', False):
    #     os.makedirs(params['tiff_output_folder'], exist_ok=True)
   
    # Iterate through layers
    for layer in range(params['layers']):
        for det in range(params['dets']):
            params['output_file'] = params['base_dir'] + '{}_layer{}_det{}.h5'.format(params['sample_name'], layer, det)
            
            # Use self.bg_pct to dynamically set the "bg" part in the filename
            params['bgsub_h5_file'] = params['base_dir'] + '{}_layer{}_det{}_{}bg.h5'.format(params['sample_name'], layer, det, params['bg_pct'])

            start_num = params['start_constant'] + layer*params['num_images']*params['omega']*params['dets'] + det*params['num_images']*params['omega']
            end_num = start_num + params['num_images']*params['omega'] -1

            print("Currently executing for layer: {}, det: {}, start_num: {}".format(layer, det, start_num))  # print current layer, det and start_num

            params['start_num'] = start_num
            params['end_num'] = end_num

        # Call conversion function with the specific converter class you want to use
            if args.command == 'slice':
                run_function(SliceHDF5, **params)
            elif args.command == 'check_file':
                run_function(Check_file_info, **params)
            elif args.command == 'rm_mask':
                run_function(Removal_Mask, **params)
            elif args.command == 'int_images':
                run_function(Integrate_images_in_hdf5, **params)            
            elif args.command == 'stand':
                run_function(Standardize_format, **params)
            elif args.command == 'sub':
                run_function(Subtract_background,  **params)
            elif args.command == 'ilastik':
                run_function(Process_with_ilastik, **params)
            elif args.command == 'hedm_formats':
                run_function(Convert_to_hedm_formats, **params)
            elif args.command == 'ff_HEDM_process':
                run_function(ff_HEDM_process, **params)
            elif args.command == 'all':
                run_function(Standardize_format, **params)
                run_function(Subtract_background,  **params)
                run_function(Process_with_ilastik, **params)
                run_function(Convert_to_hedm_formats, **params)
                run_function(ff_HEDM_process, **params)
            elif args.command == 'vis_diff':
                run_function(Visualize_Diff, **params)
if __name__ == "__main__":
    main()

