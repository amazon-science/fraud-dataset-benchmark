import os

from setuptools import find_packages, setup


setup(
    name="fraud_dataset_benchmark",
    version="1.0",
    
    # declare your packages
    packages=find_packages(where="src", exclude=("test",)),
    package_dir={"": "src"},
    include_package_data=True,

    # Enable build-time format checking
    check_format=False,

    # Enable type checking
    test_mypy=False,

    # Enable linting at build time
    test_flake8=False,

)
