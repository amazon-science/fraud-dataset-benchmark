import os
from glob import glob

from setuptools import find_packages, setup


setup(
    name='fraud_dataset_benchmark',
    version='1.0',
    
    # declare your packages
    packages=find_packages(where='src', exclude=('test',)),
    package_dir={'': 'src'},
    include_package_data=True,
    data_files=[('.',[
        'src/fdb/versioned_datasets/ipblock/20220607.zip',
    ])],

    # Enable build-time format checking
    check_format=False,

    # Enable type checking
    test_mypy=False,

    # Enable linting at build time
    test_flake8=False,

    # exclude_package_data={
    #     '': glob('fdb/*/__pycache__', recursive=True),
    # }
)
