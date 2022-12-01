from setuptools import setup

import codecs
import os.path
from pathlib import Path

def read(rel_path):
    here = os.path.abspath(os.path.dirname(__file__))
    with codecs.open(os.path.join(here, rel_path), 'r') as fp:
        return fp.read()

def get_version(rel_path):
    for line in read(rel_path).splitlines():
        if line.startswith('__version__'):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    else:
        raise RuntimeError("Unable to find version string.")

# read the contents of your README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(
    name = 'mltu',
    version = get_version("mltu/__init__.py"),
    long_description = long_description,
    long_description_content_type = 'text/markdown',
    url='https://pylessons.com/',
    author='PyLessons',
    author_email='pythonlessons0@gmail.com',
)