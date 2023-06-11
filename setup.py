### setup.py

```python
from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='hedm_pre',
    version='0.0.1',
    author='Ye Tian',
    author_email='ytian37@jhu.edu',
    description='Tools for pre-processing data for High Energy Diffraction Microscopy (HEDM) analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/minety/hedm_pre',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD 3-Clause',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.8',
    install_requires=[
        'numpy',
        'hexrd',
        'h5py',
        'tifffile',
        'PyYAML',
        'os',
        'h5py',
        're',
        'subprocess',
        'sys'
    ]
)
