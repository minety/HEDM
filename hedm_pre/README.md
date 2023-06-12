# HEDM Pre-processing

This Python package provides tools for pre-processing data for High Energy Diffraction Microscopy (HEDM) analysis.

## Features

- Conversion between different file formats (.ge, .tiff, .hdf5)
- Background subtraction
- Processing with ilastik
- Conversion to HEDM formats

## Installation

Before installing this package, you need to install hexrd first. Follow the link for the guide, https://github.com/HEXRD/hexrd.

Then you need to downgrade numpy to version: numpy-1.22.4:

pip install numpy==1.22.4

Use pip to install this package:

pip install hedm_pre

## Usage

The package provides several classes, including `FileConverter`, `Standardize_format`, `Subtract_background`, `Process_with_ilastik`, and `Convert_to_hedm_formats`. 

Each class can be instantiated with a dictionary of parameters and then the `convert` method can be called to perform the operation. 

Here is a basic example:

```python
from hedm_pre import FileConverter

params = {
    'input_file': 'input.tif',
    'output_file': 'output.h5',
    'bgsub': True,
    ...
}

converter = FileConverter(**params)
converter.convert()
```

For more detailed usage, please refer to the source code and comments.

License
hedm_pre is distributed under the terms of the BSD 3-Clause license. All new contributions must be made under this license.

Contact
For any question, please contact ytian37@jhu.edu.

