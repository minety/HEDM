# HEDM-Platform

HEDM-Platform represents a holistic and integrated framework designed to consolidate the finest HEDM resources available globally. As an all-encompassing platform, its primary goal is to offer a seamless workflow, encompassing the pre-processing, intermediate processing, and post-processing stages of HEDM data. Crafted with the insights of seasoned professionals, HEDM-Platform addresses the disparities in data standards across various synchrotron radiation sources, including APS, CHESS, SOLEIL, DESY, and more. Beyond just being a toolkit, it serves as a unified platform that empowers users to juxtapose and discern the merits and demerits of predominant software in the field. A standout feature of the platform is its embrace of AI capabilities, leveraging tools like ilastik, with an eye on future integrations and expansions into deep learning realms. This strategic incorporation of AI is pivotal in addressing intricate challenges, especially when processing data marked by pronounced strain, streak-like patterns, and substantial overlaps.

Currently, the platform amalgamates and builds upon various open-source software, including HEXRD, ImageD11, HEXOMAP, and ilastik. The corresponding links are:
- [HEXRD](https://github.com/HEXRD)
- [ImageD11](https://github.com/FABLE-3DXRD/ImageD11)
- [HEXOMAP](https://github.com/HeLiuCMU/HEXOMAP)
- [ilastik](https://www.ilastik.org/)

## Prerequisites
Before installing HEDM-Platform, you must install HEXRD, ImageD11, hdf5plugin and ilastik (advanced usage; optional, can be skipped if not needed). For detailed information, please refer to the respective URLs provided above. It's recommended to create a conda environment, e.g., `HEDM-Platform`.

First, create the HEDM-Platform conda environment using the following command:

```bash
conda create --name HEDM-Platform
conda activate HEDM-Platform
```

Next, you can install the required software using the following steps:

1. Install HEXRD:
   ```bash
   conda install -c hexrd -c conda-forge hexrd
   ```

2. Install ImageD11:
   ```bash
   python -m pip install ImageD11
   ```

3. Install hdf5plugin:
   ```bash
   pip install hdf5plugin
   ```

4. For ilastik installation: Please refer to its [official website](https://www.ilastik.org/) for detailed installation instructions.

## Installing HEDM-Platform
You can easily install the HEDM-Platform using pip:
```
pip install HEDM-Platform
```

## Getting Started
### Demo Setup
To begin with the platform, first, create a working directory where you intend to process your data. This should be an empty folder dedicated to subsequent data processing. Once you've set up your working directory, navigate to it and execute the `copy_demo` command within the `HEDM-Platform` environment. Upon execution, you'll notice the addition of several files in your directory, including:
```
config.yml
nugget_2_frames_for_test.h5
nugget_layer0_det0_for_test.flt
nugget_layer0_det0.par
```
**Note**: Due to the need for manual completion of the ilastik project file within the ilastik visual interface, and the large size of the project file making it unsuitable for packaging within this platform, users are required to export it on their own and place it in the appropriate directory. Make sure to update the necessary paths in the `config.yml` file accordingly.

### Testing Main Features

1. **Standardizing Original Files**:

  Firstly, modify the `config.yml` file as shown below before executing the subsequent command:

  <img src="https://raw.githubusercontent.com/HurleyGroup/HEDM-Platform/main/HEDM_Platform/HEDM_Platform/data/stand.jpg" alt="stand" width="800"/>

  - **Area 1**: This represents the working directory, which should have the same name as the previously copied `copy_demo` folder.
  
  - **Area 2**: This signifies the prefix of the file names. For the 'demo', there's no need to change this. However, users can adjust it based on their requirements for other files.
  
  - **Area 3**: Represents the file names that require standardization. Currently, the supported formats are .ge and .h5 files.
  
  - **Area 4**: These are the empty frames. Often, many beamtimes include some empty frames to prevent data misalignment issues when the motor begins to rotate.
  
  - **Area 5**: This denotes the total number of frames in the entire dataset, which includes the empty frames.

  After updating these five sections, run the following command to standardize the data:

  ```
  hedm-platform stand config.yml
  ```

2. **Background Noise Reduction on Standardized Files**:

   Begin by modifying the `config.yml` file as illustrated below:

   <img src="https://raw.githubusercontent.com/HurleyGroup/HEDM-Platform/main/HEDM_Platform/HEDM_Platform/data/sub.jpg" alt="sub" width="800"/>

  - **Area 1**: This represents the step size of the rotation angle.
  
  - **Area 2**: Denotes the number of frames used for the percentile filter. Typically, this count should cover at least a quarter of the rotation, equivalent to 90 degrees.
  
  - **Area 3**: This is the specific percentile value. While the standard is set at 50, in instances where there's excessive noise or when spots overlap considerably, this might need adjustment to 75%, or even 95% in some cases. The precise value to be used should be determined based on the specific data set.
  
  - **Area 4**: This introduces the option to flip the data. Often, the pattern viewed from the upstream during data collection might be flipped compared to the pattern collected by the detector. In such cases, adjustments are needed. Seven flipping scenarios are provided for selection. It's recommended to consult with the beamline scientist for correct calibration.

  Once the necessary modifications have been made, execute the following command to reduce background noise:

  ```
  hedm-platform sub config.yml
  ```

3. **Machine Learning Processing**:

   Process the noise-reduced files through machine learning to extract key spots information. Note: Manual extraction of *.ilp files from ilastik output is necessary.

   Begin by modifying the `config.yml` file as illustrated below:

   <img src="https://raw.githubusercontent.com/HurleyGroup/HEDM-Platform/main/HEDM_Platform/HEDM_Platform/data/ilastik.jpg" alt="ilastik" width="800"/>

  - **Area 1**: This represents the executable path for ilastik. For more details on its installation and path retrieval, refer to the official ilastik website's installation guide.

  - **Area 2**: Represents the .ilp project file generated within ilastik. The method to produce this will be detailed further below.
  
  - **Area 3**: If you decide to bypass ilastik's machine learning data processing, you can directly input the file path and name here. This will instruct the system to use the specified file for subsequent steps by default. If you opt for processing through ilastik, you should comment out this section.

   To generate the ilastik project file:
   - Launch ilastik and import the sliced file into the pixel clarification section. For project file creation, you can manually train the data using several dozen frames.
   - In the feature selection section (shown in the image on the left), ensure you set it to 2D. This step is crucial to avoid excessive computational overhead during subsequent processing.
   - Finally, refer to the image on the right to configure spots and background settings. Engage in manual adjustments and training. Ensure you follow the sequence shown in the image.

   <img src="https://raw.githubusercontent.com/HurleyGroup/HEDM-Platform/main/HEDM_Platform/HEDM_Platform/data/ilastik_a.jpg" alt="ilastik_a" width="550"/> <img src="https://raw.githubusercontent.com/HurleyGroup/HEDM-Platform/main/HEDM_Platform/HEDM_Platform/data/ilastik_b.jpg" alt="ilastik_b" width="180"/>

   Once you have made the necessary adjustments, execute the following command:

   ```
   hedm-platform ilastik config.yml
   ```
  
4. **Format Conversion**:

   Convert files obtained from the two previous steps (either Background Noise Reduction or Machine Learning Processing) to formats suitable for subsequent HEDM software import and calibration.

   Begin by modifying the `config.yml` file as illustrated below:

   <img src="https://raw.githubusercontent.com/HurleyGroup/HEDM-Platform/main/HEDM_Platform/HEDM_Platform/data/hedm_formats.jpg" alt="hedm_formats" width="800"/>

  - **Red Highlighted Area**: If set, this will bypass the ilastik processing and directly convert the standardized files to HEDM-related formats. If you've opted to process using the ilastik step, you should comment out this section. The software will then use the default pipeline, eliminating the need for users to input file names manually.

   After making the necessary adjustments, execute the following command:

   ```
   hedm-platform hedm_formats config.yml
   ```

5. **Preparation for testing Indexing Grains**: As the original files are quite large, previous steps utilized a slice file with only 2 frames for testing. Depending on whether the ilastik step was executed, there are two scenarios:

  - If you executed the ilastik step (Machine Learning Processing), rename the `nugget_layer0_det0_for_test.flt` to `nugget_layer0_det0_50bg_ilastik_proc_t1.flt` and replace the existing file.
  - If you did not execute the ilastik step and used the noise-reduced files directly for the subsequent indexing step, rename the `nugget_layer0_det0_for_test.flt` to `nugget_layer0_det0_50bg_t1.flt` and replace the existing file.

6. **Index Grains and Fit**:

   This test is designed to locate grains and fit them to retrieve essential details such as the strain tensor, position, orientation, and more.

   Begin by modifying the `config.yml` file as illustrated below:

   <img src="https://raw.githubusercontent.com/HurleyGroup/HEDM-Platform/main/HEDM_Platform/HEDM_Platform/data/ff_HEDM_process.jpg" alt="ff_HEDM_process" width="800"/>

  - **Area 1**: This signifies the crystal structure.
  
  - **Area 2**: Defines the range of hkl to search for (note that counting starts from 0).
  
  - **Area 3**: Represents the total number of peaks and the distinct peaks.
  
  - **Area 4**: Specifies the decreasing tolerance sequence during the search process.
  
  - **Area 5**: This is the minimum tolerance for the distance between two grains (measured in micrometers).
  
  - **Area 6**: Refers to the constant background subtraction when generating the .flt file. Note that if the data has been processed by ilastik, the intensity value ranges from 0 to 100, corresponding to the probability of spot detection. If the data hasn't been processed by ilastik, users might need to review the intensity value distribution in the h5 file before generating the .flt file to determine the appropriate value.
  
  - **Area 7**: This is the file name prefix.
  
  - **Area 8**: Specifies the tolerance for grain orientation angles.
  
  - **Area 9**: Defines the search range, represented as the radius of the search cylinder (measured in micrometers).
  
  - **Area 10**: Represents the height of the search range (measured in micrometers).
  
  - **Area 11**: Indicates the number of CPUs used for parallel processing. If you wish to utilize all available CPUs, comment out this section.

   After making the necessary adjustments, execute the following command:

   ```
   hedm-platform ff_HEDM_process config.yml
   ```

### Additional Useful Features
(Description of other features can be added here)

---

For any questions, please contact ytian37@jhu.edu/ytian6688@hotmail.com or rhurley6@jhu.edu.

License: HEDM-Platform is distributed under the terms of the BSD 3-Clause license. All new contributions must be made under this license.

This software package was predominantly developed by the author, Ye Tian, within Prof. Ryan Hurley's research group at Johns Hopkins University (JHU). Acknowledgments are also due for certain contributions made during the author's affiliation with Prof. Todd Hufnagel's research group.

---