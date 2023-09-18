from setuptools import setup, find_packages

with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='HEDM_Toolkit',
    version='0.3.1',
    author='Ye Tian',
    author_email='ytian37@jhu.edu',
    entry_points={
        'console_scripts': [
            'HEDM_Toolkit=HEDM_Toolkit.HEDM_Toolkit:main',
            'copy_demo=HEDM_Toolkit.utilities:copy_demo_func', # assuming the function is in utilities module
        ],
    },
    description='Tools for High Energy Diffraction Microscopy (HEDM) analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/minety/HEDM',
    packages=find_packages(),
    package_data={
        'HEDM_Toolkit': ['scripts/*', 'data/*', 'Demo/*'],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: BSD License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.8'
)
