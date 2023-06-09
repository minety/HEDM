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

class FileConverter:
    def __init__(self, **kwargs):
        self.params = kwargs
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
        self.use_parallel_computing = self.params.get('use_parallel_computing', False)

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

    def ge2hdf5(self):
        # width and height for the images (Default is 2048*2048 for most beamlines)
        width = 2048
        height = 2048

        # Read raw data from ge5 file
        with open(self.input_file, 'rb') as f:
            # 8192 was found to be the header (skip this part) APS sector1
            f.read(8192)

            # Read all the images as numpy array
            data = np.fromfile(f, dtype=np.uint16)

        # Convert all binary data to an array of images.
        # image_data = np.reshape(data, (self.num_images, height, width))

        try:
            image_data = np.reshape(data, (self.num_images, height, width))
        except ValueError:
            raise ValueError("Please input the right total number of frames: num_images")


        # Remove the specified number of empty images
        if self.empty_images > 0:
            image_data = image_data[self.empty_images:]

        # Create an HDF5 file
        with h5py.File(self.output_file, 'w') as f:
            # Create a dataset with the shape (num_images-empty_images, height, width) and the same dtype as the first image
            dataset = f.create_dataset('images', (len(image_data), height, width), dtype=image_data.dtype)
            # Print the number of images in the dataset
            print(f"Number of images in the dataset: {len(image_data)}")
            # Write the images
            for i, img in enumerate(image_data):
                dataset[i] = img

    # May need to modified later for SOLEIL data
    def hdf5_to_hdf5(self):
        # Copy the input HDF5 file to the output file with a new name and change the internal path to /images
        with h5py.File(self.input_file, 'r') as input_file:
            with h5py.File(self.output_file, 'w') as output_file:
                # Iterate over all keys in the input file
                for key in input_file.keys():
                    # Get the dataset from the input file
                    input_dataset = input_file[key]

                    # Remove the specified number of empty images
                    if self.empty_images > 0:
                        input_dataset = input_dataset[self.empty_images:]

                    # Create a new dataset in the output file with the same shape and dtype as the input dataset
                    output_dataset = output_file.create_dataset(f'/images', shape=input_dataset.shape, dtype=input_dataset.dtype)
                    # Copy the data from the input dataset to the output dataset
                    output_dataset[()] = input_dataset[()]

    def bgsub4hdf5(self):
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

        # Make dark image from first 100 frames
        pct = self.bg_pct
        nf_to_use = self.bg_nf
        dark = imageseries.stats.percentile(ims, pct, nf_to_use)
        # np.save(DarkFile, dark)

        # Now, apply the processing options
        ProcessedIS = imageseries.process.ProcessedImageSeries
        ops = [('dark', dark), ('flip', None)] # None, 'h', 'v', etc.
        pimgs = ProcessedIS(ims, ops)

        # Save the processed imageseries in hdf5 format
        imageseries.write(pimgs, self.bgsub_h5,'hdf5', path='/imageseries')

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
    
    def hdf5_to_npz(self, flip_option):
        base_name, extension = os.path.splitext(self.bgsub_h5)
        input_file = f"{base_name}_ilastik_proc{extension}"
        ims = imageseries.open(
            input_file,
            format='hdf5',
            path='/',
            # dataname='exported_data'
            dataname='images' 
        )
        # Input of omega meta data
        nf = self.bg_nf  #720
        omega = self.omega
        omw = OmegaWedges(nf)
        omw.addwedge(0, nf*omega, nf) 
        ims.metadata['omega'] = omw.omegas

        # Make dark image from first 100 frames
        pct = self.bg_pct
        nf_to_use = self.bg_nf
        dark = imageseries.stats.percentile(ims, pct, nf_to_use)
        # np.save(DarkFile, dark)

        # Now, apply the processing options
        ProcessedIS = imageseries.process.ProcessedImageSeries
        ops = [('dark', dark), ('flip', flip_option)] # None, 'h', 'v', etc.
        pimgs = ProcessedIS(ims, ops)

        output_file = f"{base_name}_hexrd.npz"
        # Save the processed imageseries in npz format
        print(f"Writing npz file (may take a while): {output_file}")
        imageseries.write(pimgs, output_file, 'frame-cache', threshold=5, cache_file=output_file)

    
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

class Standardize_format(FileConverter):
    def convert(self):
        print("Standardizing format...")
        if self.input_format in ('.ge', '.ge2', '.ge3', '.ge5'):
            print("Converting from .ge format to .h5...")
            self.ge2hdf5()
        elif self.input_format in ('.tif', '.tiff'):
            print("Converting from .tif format to .h5...")
            self.tif2hdf5()
        elif self.input_format in ('.hdf5', '.h5'):
            print("Input file is already in .h5 format. Copying to output file with a new name after deleting empty frames...")
            self.hdf5_to_hdf5()
        else:
            print(f"Unsupported input format: {self.input_format}")
        if self.slice_file:
            print("Generating sliced raw data...")
            self.hdf5slice(self.output_file, self.image_default_path)

class Subtract_background(FileConverter):
    def convert(self):
        if self.bgsub:
            print("Applying background subtraction...")
            self.bgsub4hdf5()
        if self.slice_file:
            print("Generating sliced bgsub data...")
            self.hdf5slice(self.bgsub_h5, self.imageseries_path)

class Process_with_ilastik(FileConverter):
    def convert(self):
        print(f'ilastik_project_file is {self.ilastik_project_file}')
        if self.ilastik_proc:
            print("Processing with ilastik...")
            base_name, extension = os.path.splitext(self.bgsub_h5) 
            output_file = f"{base_name}_ilastik_proc{extension}"
            
            if self.use_parallel_computing:     # If parallel computing is set to True in the config
                print("Running in parallel mode...")
                mpiexec_command = [
                    "mpirun", "-n", "12",
                    self.ilastik_loc,
                    "--headless",
                    "--distributed",
                    "--distributed-block-roi", '{"x": 2048, "y": 2048, "z": 1, "c": 2}',
                    f"--cutout_subregion=[(0,0,0,0),({self.bg_nf},2048,2048,1)]",
                    "--pipeline_result_drange=(0.0,1.0)",
                    "--export_drange=(0,100)",
                    f"--output_filename_format={output_file}",
                    f"--project={self.ilastik_project_file}",
                    "--export_source=Probabilities",
                    "--raw_data=" + f"{self.bgsub_h5}"
                ]
                subprocess.run(mpiexec_command)
            else:  # Otherwise, run the non-parallel code
                subprocess.run([
                    self.ilastik_loc,
                    '--headless',
                    f'--cutout_subregion=[(0,0,0,0),({self.bg_nf},2048,2048,1)]',
                    '--pipeline_result_drange=(0.0,1.0)',
                    '--export_drange=(0,100)',
                    f'--output_filename_format={output_file}',
                    f'--project={self.ilastik_project_file}',
                    '--export_source=Probabilities',
                    '--raw_data='+f'{self.bgsub_h5}'
                ], env={'LAZYFLOW_THREADS': '6'})
            
            self.standard_hdf5(output_file, self.image_ilastik_path, self.image_default_path)
    
        if self.slice_file:
            print("Generating sliced ilastik processed data...")
            self.hdf5slice(output_file, self.image_default_path)

class Convert_to_hedm_formats(FileConverter):
    def convert(self):
        if self.generate_hexomap_files:
            print("Generating hexomap .tif files...")
            self.hdf5_to_tiff()
        elif self.generate_hexrd_files:
            print("Generating hexrd .npz files...")
            self.hdf5_to_npz(self.flip_option)  # Replace with the actual method to generate hexrd files
            print(f"HEXRD flip option is '{self.flip_option}'.")
        elif self.generate_ImageD11_files:
            print("Generating ImageD11 .flt files...")
            self.another_method()  # Replace with the actual method to generate ImageD11 files
        else:
            print("None of the generate flags is set to True. No action will be performed.")

def run_conversion(converter_class, **params):
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
    parser.add_argument('command', choices=['stand', 'sub', 'ilastik', 'hedm_formats', 'all'])
    parser.add_argument('config_file')
    args = parser.parse_args()

    if len(sys.argv) != 3:
        print('Usage: python HEDM_Pre.py <command> <config_file>')
        sys.exit()

    # Load configuration
    config_file = args.config_file
    params = load_config(config_file)
    if params is None:
        print('Error loading configuration')
        sys.exit()

    # Calculate bg_nf and add it to the params
    params['bg_nf'] = params['num_images'] - params['empty_images']

    # Create output directory
    os.makedirs(params['tiff_output_folder'], exist_ok=True)

    # Iterate through layers
    for layer in range(params['layers']):
        for det in range(params['dets']):
            params['output_file'] = params['base_dir'] + 'nugget1_ff_layer{}_det{}.h5'.format(layer, det)
            params['bgsub_h5_file'] = params['base_dir'] + 'nugget1_nf_layer{}_det{}_50bg.h5'.format(layer, det)

            start_num = params['start_constant'] + layer*params['num_images']*params['omega']*params['dets'] + det*params['num_images']*params['omega']
            end_num = start_num + params['num_images']*params['omega'] -1

            print("Currently executing for layer: {}, det: {}, start_num: {}".format(layer, det, start_num))  # print current layer, det and start_num

            params['start_num'] = start_num
            params['end_num'] = end_num

            # Call conversion function with the specific converter class you want to use
            if args.command == 'stand':
                run_conversion(Standardize_format, **params)
            elif args.command == 'sub':
                run_conversion(Subtract_background,  **params)
            elif args.command == 'ilastik':
                run_conversion(Process_with_ilastik, **params)
            elif args.command == 'hedm_formats':
                run_conversion(Convert_to_hedm_formats, **params)
            elif args.command == 'all':
                run_conversion(Standardize_format, **params)
                run_conversion(Subtract_background,  **params)
                run_conversion(Process_with_ilastik, **params)
                run_conversion(Convert_to_hedm_formats, **params)

if __name__ == "__main__":
    main()

