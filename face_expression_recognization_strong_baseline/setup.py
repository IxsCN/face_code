import setuptools, os
from setuptools import  find_packages

PACKAGE_NAME = 'fer_strong_baseline'
VERSION = '0.1.0'
AUTHOR = 'stoneyang159'
EMAIL = 'stoneyang159@gmail.com'
DESCRIPTION = 'Face Expression Recognition Strong Baseline'
GITHUB_URL = 'https://github.com/stoneyang159/face_expression_recognization_strong_baseline'


setuptools.setup(
    name = PACKAGE_NAME,
    version = VERSION,
    author = AUTHOR,
    author_email = EMAIL,
    description = DESCRIPTION,
    long_description="",
    long_description_content_type='text/markdown',
    url = GITHUB_URL,
    packages=find_packages(include=('fer_strong_baseline')),
    package_dir={'fer_strong_baseline':'.'},
    package_data={'': ['*net.pt']},
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[]
    
)
