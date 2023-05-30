import numpy as np
from hexrd import imageseries
from hexrd.imageseries.omega import OmegaWedges
import os
import h5py
import re
from tifffile import imread, TiffFile, imsave

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
        self.slice_input_file = self.params.get('slice_input_file', None)
        self.slice_output_file = self.params.get('slice_output_file', None)
    

    def get_tiff_compression(tiff_path):
        with TiffFile(tiff_path) as tif:
            return tif.pages[0].compression

    def _get_file_format(self):
        _, ext = os.path.splitext(self.input_file)
        return ext.lower()
    
    def _get_input_folder(self):
        return os.path.dirname(self.input_file)

    def convert(self, *args, **kwargs):
        raise NotImplementedError("Subclass should implement this method")

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
        image_data = np.reshape(data, (self.num_images, height, width))

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

    def hdf5_to_tiff(self, output_offset, output_tiff_folder, prefix, input_tiff_folder, start_num=None, end_num=None):
        with h5py.File(input_proc_file, 'r') as f:
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
            compression = self.get_tiff_compression(first_tiff_path)

            for i in range(start_idx, end_idx):
                img = dataset[i].squeeze()
                img[img < 50] = 0
                img_filename = f"{prefix}_{i + output_offset:06d}.tif"
                img_path = os.path.join(output_tiff_folder, img_filename)
                imsave(img_path, img, compression=compression)

    def hdf5slice(self):
        with h5py.File(self.slice_input_file, 'r') as f:
            images = f[self.image_slice_path]
            with h5py.File(self.slice_output_file, 'w') as f_example:
                example_images = f_example.create_dataset('images', (self.slice_images, images.shape[1], images.shape[2]), dtype=images.dtype)
                example_images[:] = images[:self.slice_images, :, :]

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
           self.slice_input_file = self.output_file
           self.image_slice_path = self.image_default_path
           self.hdf5slice()

class Subtract_background(FileConverter):
    def convert(self):
        if self.bgsub:
            print("Applying background subtraction...")
            self.bgsub4hdf5()

class Process_with_ilastik(FileConverter):
    def convert(self):
        if self.ilastik_proc:
            print("Processing with ilastik...")
            self.ilastik_proc_h5() 

class Convert_to_hedm_formats(FileConverter):
    def convert(self):
        if self.generate_hexomap_files:
            print("Generating hexomap files...")
            self.hdf5_to_tiff()

def run_conversion(converter_class, **params):
    converter = converter_class(**params)
    converter.convert()

if __name__ == "__main__":
    base_dir = '/Users/yetian/Desktop/Ryan_test_data/APS_2023Feb/'
    # input_file = base_dir + 'nugget1_nf_int_before/nugget1_nf_int4_000000.tif'
    input_file = base_dir + 'test_nugget1_002_000050.edf.ge5'
    image_input_path = '/input_path'
    slice_file = True
    slice_images = 1
    empty_images = 2
    num_images = 200 # number of frames in total in hdf5 file
    omega = 0.1 # omega rotation angle in degree
    bg_pct = 50 # background to subtract in percentile
    bg_nf = num_images-empty_images # number of frames to use to generate dark field image   
    bgsub = True
    start_constant = 0 # replace with the constant you mentioned

    layers = 1 # replace with the actual value
    dets = 1 # replace with the actual value

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
                'slice_file': slice_file,
                'slice_input_file': bgsub_h5_file if bgsub else output_file,
                'slice_output_file': slice_output_file,
                'slice_images': slice_images
            }

    run_conversion(Standardize_format, **params)


