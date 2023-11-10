from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='HEDM_Platform',
    version='0.3.9',
    author='Ye Tian',
    author_email='ytian6688@hotmail.com, ytian37@jhu.edu',
    entry_points={
        'console_scripts': [
            'hedm-platform=HEDM_Platform.HEDM_Platform:main',
            'copy_demo=HEDM_Platform.utilities:copy_demo_func', 
        ],
    },
    description='Platform for High Energy Diffraction Microscopy (HEDM) Analysis, also known as 3D X-ray Diffraction (3DXRD)',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/HurleyGroup/HEDM-Platform/tree/main/HEDM_Platform',
    packages=find_packages(),
    package_data={
        'HEDM_Platform': ['scripts/*', 'data/*', 'Demo/*'],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    python_requires='>=3.8'
)
