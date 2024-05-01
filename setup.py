from setuptools import find_packages, setup

config = {
    "name": "bird_repo",
    "version": "0.0.1",
    "author": "qraquet",
    "author_email": "qraquet@expediagroup.com",
    "classifiers": [
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    "packages": find_packages(),
    "python_requires": ">=3.8",
}

setup(**config)
