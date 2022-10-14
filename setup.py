import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

VERSION = '0.1.0'
PACKAGE_NAME = 'wildboottest'
AUTHOR = ['Alexander Fischer', 'Aleksandr Michuda']
AUTHOR_EMAIL = ['alexander-fischer1801@t-online.de', 'amichuda@gmail.com']
URL = 'https://github.com/s3alfisc/wildboottest'

LICENSE = 'MIT'
DESCRIPTION = 'Wild Cluster Bootstrap Inference for Linear Models in Python'
LONG_DESCRIPTION = (HERE / "readme.md").read_text()
LONG_DESC_TYPE = "text/markdown"

INSTALL_REQUIRES = [
      'numpy',
      'pandas', 
      'numba'
]

setup(name=PACKAGE_NAME,
      version=VERSION,
      description=DESCRIPTION,
      long_description=LONG_DESCRIPTION,
      long_description_content_type=LONG_DESC_TYPE,
      author=AUTHOR,
      license=LICENSE,
      author_email=AUTHOR_EMAIL,
      url=URL,
      install_requires=INSTALL_REQUIRES,
      packages=find_packages()
      )
