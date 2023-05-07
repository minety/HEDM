import numpy as np
from hexrd import imageseries
from hexrd.imageseries.omega import OmegaWedges
import os
import h5py
import re
from tifffile import imread


class FileConverter:
    def __init__(self, **kwargs):
        self.params = kwargs
        self.input_file = self.params.get('input_file')
        self.output_file = self.params.get('output_file')
        self.input_format = self._get_file_format()
        self.input_folder = self._get_input_folder()
        self.start_num = self.params.get('start_num', None)
        self.end_num = self.params.get('end_num', None) 
        # self.gefile = self.params.get('gefile', None)
        self.num_images = self.params.get('num_images', None)
        self.omega = self.params.get('omega', None)
        self.bg_pct = self.params.get('bg_pct', None)
        self.bg_nf = self.params.get('bg_nf', None) 
        self.bgsub_h5 = self.params.get('bgsub_h5_file', None)
        self.bgsub = self.params.get('bgsub', True)
        self.empty_images = self.params.get('empty_images', 0)


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
            raise ValueError("No image paths found with the given prefix.")

        # Sort the image paths to ensure they are in the correct order
        image_paths = sorted(image_paths, key=lambda x: int(re.search(r'\d{6}', x).group()))

        if self.start_num is not None and self.end_num is not None:
            image_paths = image_paths[self.start_num:self.end_num+1]

        # Load the first image to get dimensions
        first_image = imread(image_paths[0])
        h, w = first_image.shape

        # Create an HDF5 file
        with h5py.File(self.output_file, 'w') as f:
            # Create a dataset with the shape (num_images, height, width) and the same dtype as the first image
            dataset = f.create_dataset('images', (len(image_paths), h, w), dtype=first_image.dtype)
            # Print the number of images in the dataset
            print(f"Number of images in the dataset: {len(image_paths)}")
            # Write the first image
            dataset[0] = first_image

            # Write the remaining images
            for i, img_path in enumerate(image_paths[1:], start=1):
                img = imread(img_path)
                dataset[i] = img

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
            # Create a dataset with the shape (num_images, height, width) and the same dtype as the first image
            dataset = f.create_dataset('images', (self.num_images, height, width), dtype=image_data.dtype)
            # Print the number of images in the dataset
            print(f"Number of images in the dataset: {self.num_images}")
            # Write the images
            for i, img in enumerate(image_data):
                dataset[i] = img

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

class ToIlastikConverter(FileConverter):
    def convert(self):
        print("Converting to ilastik format...")
        if self.input_format in ('.ge', '.ge2', '.ge3', '.ge5'):
            self.ge2hdf5()
        elif self.input_format in ('.tif', '.tiff'):
            print("Converting from .tif format to .h5...")
            self.tif2hdf5()
        elif self.input_format in ('.hdf5', '.h5'):
            if not self.bgsub:
                print("Input file is already in .h5 format. Copying to output file with a new name...")
                self.hdf5_to_hdf5()
            # If bgsub is True, proceed with the background subtraction operation below
        else:
            print(f"Unsupported input format: {self.input_format}")

        if self.bgsub:
            print("Applying background subtraction...")
            self.bgsub4hdf5()


class ToHexrdConverter(FileConverter):
    def convert(self):
        print("Converting to hexrd format...")
        # Code to convert to hexrd format
        pass

class ToHexomapConverter(FileConverter):
    def convert(self):
        print("Converting to hexomap format...")
        # Code to convert to hexomap format
        pass

class ToImageD11Converter(FileConverter):
    def convert(self):
        print("Converting to ImageD11 format...")
        # Code to convert to ImageD11 format
        pass

def run_conversion(converter_class, **params):
    converter = converter_class(**params)
    converter.convert()

if __name__ == "__main__":
    # Example usage, for .tif/.tiff files just choose a random (number) file with the right prefix you want to process
    input_file = '/Users/yetian/Dropbox/My Mac (Ye’s MacBook Pro)/Desktop/Ryan_test_data/APS_2023Feb/nf_test/class_test_1.h5'
    output_file = '/Users/yetian/Desktop/Ryan_test_data/APS_2023Feb/nf_test/class_test_1_1.h5'
    empty_images = 2
    start_num = 0
    end_num = 10
    bgsub_h5_file = '/Users/yetian/Desktop/Ryan_test_data/APS_2023Feb/nf_test/class_bgsub_50pct_test_1.h5'
    num_images = 80 # number of frames in total in hdf5 file
    omega = 0.25 # omega rotation angle in degree
    bg_pct = 75 # background to subtract in percentile
    bg_nf = 30 # number of frames to use to generate dark field image   
    bgsub = False

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
        'bgsub': bgsub
    }
    
    run_conversion(ToIlastikConverter, **params)
    


