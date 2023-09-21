# HEDM-Platform

HEDM-Platform represents a holistic and integrated framework designed to consolidate the finest HEDM resources available globally. As an all-encompassing platform, its primary goal is to offer a seamless workflow, encompassing the pre-processing, intermediate processing, and post-processing stages of HEDM data. Crafted with the insights of seasoned professionals, HEDM-Platform addresses the disparities in data standards across various synchrotron radiation sources, including APS, CHESS, SOLEIL, DESY, and more. Beyond just being a toolkit, it serves as a unified platform that empowers users to juxtapose and discern the merits and demerits of predominant software in the field. A standout feature of the platform is its embrace of AI capabilities, leveraging tools like ilastik, with an eye on future integrations and expansions into deep learning realms. This strategic incorporation of AI is pivotal in addressing intricate challenges, especially when processing data marked by pronounced strain, streak-like patterns, and substantial overlaps.

Currently, the platform amalgamates and builds upon various open-source software, including HEXRD, ImageD11, HEXOMAP, and ilastik. The corresponding links are:
- [hdf5plugin](https://pypi.org/project/hdf5plugin/)
- [HEXRD](https://github.com/HEXRD)
- [ImageD11](https://github.com/FABLE-3DXRD/ImageD11)
- [HEXOMAP](https://github.com/HeLiuCMU/HEXOMAP)
- [ilastik](https://www.ilastik.org/)

## Prerequisites
Before installing HEDM-Platform, you must install hdf5plugin, HEXRD, ImageD11, and ilastik (advanced usage; optional, can be skipped if not needed). For installation instructions, please refer to the respective URLs provided above. It's recommended to create a conda environment, e.g., `HEDM-Platform`.

First, create the HEDM-Platform conda environment using the following command:

```bash
conda create --name HEDM_Platform python=3.8
conda activate HEDM_Platform
```

Next, you can install the required software using the following steps:

1. Install hdf5plugin:
   ```bash
   pip install hdf5plugin
   ```

2. Install HEXRD:
   ```bash
   conda install -c hexrd -c conda-forge hexrd
   ```

3. Install ImageD11:
   ```bash
   python -m pip install ImageD11
   ```

4. For ilastik installation: Please refer to its [official website](https://www.ilastik.org/) for detailed installation instructions.

## Installing HEDM-Platform
You can easily install the HEDM-Platform using pip:
```
pip install HEDM-Platform
```

## Getting Started
### Demo Setup
To start with the platform, first, copy the demo folder for testing purposes. Navigate to the working directory where you wish to process data and execute the `copy_demo` command within the `HEDM-Platform` environment. Upon execution, you'll notice the addition of several files in your directory, including:
```
config.yml
nugget_2_frames_for_test.h5
nugget_layer0_det0_for_test.flt
nugget_layer0_det0_50bg.par
```
**Note**: Due to the need for manual completion of the ilastik project file within the ilastik visual interface, and the large size of the project file making it unsuitable for packaging within this platform, users are required to export it on their own and place it in the appropriate directory. Necessary paths should be updated in the `config.yml` file accordingly.

### Testing Main Features
- **Standardizing Original Files**: 
  ```
  HEDM-Platform stand config.yml
  ```
  
- **Background Noise Reduction on Standardized Files**: Perform pixel-based median background noise reduction.
  ```
  HEDM-Platform sub config.yml
  ```
  
- **Machine Learning Processing**: Process the noise-reduced files through machine learning to extract key spots information. Note: Manual extraction of *.ilp files from ilastik output is required.
  ```
  HEDM-Platform ilastik config.yml
  ```
  
- **Format Conversion**: Convert files obtained from the two previous steps (either Background Noise Reduction or Machine Learning Processing) for subsequent HEDM software import and calibration.
  ```
  HEDM-Platform hedm_formats config.yml
  ```

- **Preparation for testing Indexing Grains**: As the original files are quite large, previous steps utilized a slice file with only 2 frames for testing. Depending on whether the ilastik step was executed, there are two scenarios:

  - If you executed the il

astik step (Machine Learning Processing), rename the `nugget_layer0_det0_for_test.flt` to `nugget_layer0_det0_50bg_ilastik_proc_t1.flt` and replace the existing file.
  - If you did not execute the ilastik step and used the noise-reduced files directly for the subsequent indexing step, rename the `nugget_layer0_det0_for_test.flt` to `nugget_layer0_det0_50bg_t1.flt` and replace the existing file.

- **Index Grains and Fit**: Test to find grains and fit to obtain information like strain tensor, position, orientation, etc.
  ```
  HEDM-Platform ff_HEDM_process config.yml
  ```

### Additional Useful Features
(Description of other features can be added here)

---

For any questions, please contact ytian37@jhu.edu/ytian6688@hotmail.com or rhurley6@jhu.edu.

License: HEDM-Platform is distributed under the terms of the BSD 3-Clause license. All new contributions must be made under this license.

This software package was developed within Prof. Ryan Hurley's research group at Johns Hopkins University (JHU). It also incorporates contributions developed in Prof. Todd Hufnagel's research group. Author: Ye Tian.

---
