import numpy as np
from hexrd import imageseries
from hexrd.imageseries.omega import OmegaWedges
import os
import h5py
import re
from tifffile import imread, TiffFile, imsave
import subprocess
import yaml
import sys
import argparse
import fabio
import time
import functools
import hdf5plugin
import numba
from ImageD11 import sparseframe, cImageD11, columnfile
import pkg_resources
import matplotlib.pyplot as plt
from skimage import draw
from skimage import exposure
from matplotlib.patches import Circle

class HEDM_Platform:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.base_dir = self.params['base_dir']
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
        self.ilastik_proc = self.params.get('ilastik_proc', True)
        self.ilastik_loc = self.params.get('ilastik_loc')
        self.ilastik_project_file = self.params.get('ilastik_project_file', None)
        self.generate_hexomap_files = self.params.get('generate_hexomap_files', None)
        self.tiff_path = self.params.get('tiff_path', None)
        self.prefix = self.params.get('prefix', None)
        self.tiff_output_folder = self.params.get('tiff_output_folder', None)
        self.hdf5_input_folder = self.params.get('hdf5_input_folder', None)
        self.input_tiff_folder = self.params.get('input_tiff_folder', None)
        self.selected_layers = self.params.get('selected_layers', None)
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
        self.input_file_ff = self.params.get('input_file_ff', None)
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

            # Remove the specified number of empty images if needed
            if self.empty_images > 0:
                input_dataset = input_dataset[self.empty_images:]

            # Write to the output HDF5 file
            with h5py.File(self.output_file, 'w') as output_file:
                # Create a new dataset with the same shape and dtype as the input dataset
                # Store it under the '/images' path in the output file
                output_dataset = output_file.create_dataset('/images', shape=input_dataset.shape, dtype=input_dataset.dtype)
                # Copy the data from the input dataset to the output dataset
                output_dataset[()] = input_dataset[()]


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

    def hdf5slice(self, input_file, slice_path):
        with h5py.File(input_file, 'r') as f:
            images = f[slice_path]
            base_name, extension = os.path.splitext(input_file)
            output_file = f"{base_name}_slice{extension}"
            with h5py.File(output_file, 'w') as f_example:
                example_images = f_example.create_dataset('images', (self.slice_images, images.shape[1], images.shape[2]), dtype=images.dtype)
                example_images[:] = images[:self.slice_images, :, :]

    @staticmethod 
    def get_tiff_compression(tiff_path):
        with TiffFile(tiff_path) as tif:
            return tif.pages[0].compression
         
    def sort_key(self):
        match = re.match(fr'{self.prefix}_layer(\d+)_det(\d+)_50bg_proc.h5', self.sort_filename)
        if not match:
            return float('inf')  # Place files that do not conform to the format at the end
        layer, det = map(int, match.groups())
        return layer * 2 + det  # Sort according to the numerical values of layer and det
    
    def get_start_num_from_filename(self):
        # Suppose your filename format is like 'prefix_layer1_det2_50bg_proc.h5'
        pattern = fr'nugget1_nf_layer(\d+)_det(\d+)_50bg_proc.h5'
        match = re.match(pattern, self.hdf5_file_name)
        if not match:
            raise ValueError(f"Invalid filename: {self.hdf5_file_name}")
        layer, det = map(int, match.groups())
        return (layer * 2 + det) * 180

    def hdf5_to_tiff(self):
        with h5py.File(self.hdf5_file, 'r') as f:
            dataset = f[self.image_ilastik_path]
            output_offset = self.get_start_num_from_filename()
            # The length of dataset
            num_images = len(dataset)
            # start and end idx
            start_idx = 0
            end_idx = num_images
            if start_idx >= num_images or end_idx > num_images:
                raise ValueError(f"Invalid range: start_num and end_num must be within the range of the dataset (0-{num_images - 1}).")
            input_tiff_folder = os.path.dirname(self.input_file)
            # get the tiff compression method for nf-hedm at aps
            first_tiff_path = [os.path.join(input_tiff_folder, p) for p in os.listdir(input_tiff_folder) if p.lower().endswith('.tif') or p.lower().endswith('.tiff')][0]
            compression = self.get_tiff_compression(first_tiff_path)
            start_idx = int(start_idx)
            end_idx = int(end_idx)
            for i in range(start_idx, end_idx):
                img = dataset[i].squeeze()
                img[img < 50] = 0
                img_filename = f"{self.prefix}_{i + output_offset:06d}.tif"
                img_path = os.path.join(tiff_output_folder, img_filename)
                imsave(img_path, img, compression=compression)

    def get_layer_from_filename(self):
        match = re.match(fr'{self.prefix}_layer(\d+)_det(\d+)_50bg_proc.h5', self.sort_filename)
        if not match:
            raise ValueError(f"Invalid filename: {self.sort_filename}")
        layer, det = map(int, match.groups())
        return layer

    def hdf5_to_npz(self):
        base_name, extension = os.path.splitext(self.bgsub_h5)

        # Use the provided input_file_ff or the default if not provided
        if self.input_file_ff:
            input_file_ff = self.input_file_ff
            # Remove 'ilastik_' when input_file_ff is provided
            output_file = f"{base_name}_proc.npz"
        else:
            input_file_ff = f"{base_name}_ilastik_proc{extension}"
            output_file = f"{base_name}_ilastik_proc.npz"
        
        # Check for the possible data paths in the HDF5 file
        with h5py.File(input_file_ff, 'r') as hf:
            if 'images' in hf:
                dataname = 'images'
            elif 'imageseries/images' in hf:
                dataname = 'imageseries/images'
            else:
                raise ValueError("The HDF5 file does not contain either 'images' or 'imageseries/images' data paths.")

        ims = imageseries.open(
            input_file_ff,
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
        
        # Use the provided input_file_ff or the default if not provided
        input_file_ff = self.input_file_ff if self.input_file_ff else f"{base_name}_ilastik_proc{extension}"
        
        # Open the ilastik processed file
        with h5py.File(input_file_ff, 'r') as f:
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
        # Use the provided input_file_ff or the default if not provided
        input_file_ff = self.input_file_ff if self.input_file_ff else f"{base_name}_ilastik_proc{extension}"
        bg_filename = f"{base_name}_bg.edf"
        
        # Construct the full command
        cmd = [
            'python', script_path,
            input_file_ff,
            bg_filename,
            str(self.flt_THRESHOLDS),
            str(self.num_images),
            str(self.base_dir)
        ]
        
        # Execute the command
        subprocess.run(cmd)

    def run_gen_flt_script(self):
        script_path = pkg_resources.resource_filename('HEDM_Platform', 'scripts/gen_flt.py')
        dx_file = pkg_resources.resource_filename('HEDM_Platform', 'data/imaged11_dx.edf')
        dy_file = pkg_resources.resource_filename('HEDM_Platform', 'data/imaged11_dy.edf')
        
        # Construct the input and output filenames
        base_name, extension = os.path.splitext(self.bgsub_h5)
        # Use the provided input_file_ff or the default if not provided
        if self.input_file_ff:
            input_file_ff = self.input_file_ff
            # Remove 'ilastik_' when input_file_ff is provided
            sparse_file = f"{base_name}_t{str(self.flt_THRESHOLDS)}_sparse.h5"
            flt_file = f"{base_name}_t{str(self.flt_THRESHOLDS)}.flt"
        else:
            input_file_ff = f"{base_name}_ilastik_proc{extension}"
            sparse_file = f"{base_name}_ilastik_proc_t{str(self.flt_THRESHOLDS)}_sparse.h5"
            flt_file = f"{base_name}_ilastik_proc_t{str(self.flt_THRESHOLDS)}.flt"

        par_file = base_name.replace(f"_{self.bg_pct}bg", "") + ".par"
 
        # Construct the full command
        cmd = [
            'python', script_path, 
            input_file_ff,
            sparse_file,
            flt_file,
            par_file,
            dx_file,  # Adding dx_file to the command
            dy_file   # Adding dy_file to the command
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
        if self.input_file_ff:
            flt_file_name = os.path.basename(f"{base_name}_t{str(self.flt_THRESHOLDS)}.flt")
        else:
            flt_file_name = os.path.basename(f"{base_name}_ilastik_proc_t{str(self.flt_THRESHOLDS)}.flt")        

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
        par_file = f"{base_name}.par"
        Idx_file_prefix = self.Idx_prefix
        
        # Determine the flt_file_name based on the presence of input_file_ImageD11
        if self.input_file_ff:
            flt_file_name = os.path.basename(f"{base_name}_t{str(self.flt_THRESHOLDS)}.flt")
        else:
            flt_file_name = os.path.basename(f"{base_name}_ilastik_proc_t{str(self.flt_THRESHOLDS)}.flt")

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
        par_file = f"{base_name}.par"
        # Determine the flt_file_name based on the presence of input_file_ImageD11
        if self.input_file_ff:
            flt_file_name = os.path.basename(f"{base_name}_t{str(self.flt_THRESHOLDS)}.flt")
        else:
            flt_file_name = os.path.basename(f"{base_name}_ilastik_proc_t{str(self.flt_THRESHOLDS)}.flt")
        t_values = [0.05, 0.04, 0.03, 0.02, 0.01]
        omega_slop = 0.05
        
        u_file = f'all{self.Idx_prefix}.map'
        for i, t in enumerate(t_values):
            U_file = f'all{self.Idx_prefix}_it{i+1}.map'
            self.run_ImageD11_fitting_script(u_file, U_file, par_file, flt_file_name, t, omega_slop)
            u_file = U_file

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
        elif self.input_format in ('.hdf5', '.h5', '.nxs'):  # Added .nxs here
            print("Converting from .h5/.nxs format to .h5...")  # Updated the message to include .nxs
            self.hdf5_to_hdf5()
        else:
            print(f"Unsupported input format: {self.input_format}")

class Subtract_background(HEDM_Platform):
    def convert(self):
        if self.bgsub:
            print("Applying background subtraction...")
            self.bgsub4hdf5(self.flip_option)
            print(f"HEXRD flip option is '{self.flip_option}'.")

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

        elif self.hexrd_process:
            print("Hexrd find orientation...")
            self.hexrd_process()
            pass 
        elif self.x:
            print("x...")
            self.x()
            pass
        else:
            print("x.")

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
    parser.add_argument('command', choices=['check_file', 'check_flip','rm_mask','stand', 'sub', 'ilastik', 'hedm_formats', 'all', 'slice', 'ff_HEDM_process', 'vis_diff'])
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

    # Create output directory only if generate_hexomap_files is True
    if params.get('generate_hexomap_files', False):
        os.makedirs(params['tiff_output_folder'], exist_ok=True)
   
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

