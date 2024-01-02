import os
from setuptools import setup, find_packages


DIR = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(DIR, "README.md")) as fh:
    long_description = fh.read()

with open(os.path.join(DIR, "requirements.txt")) as fh:
    requirements = fh.read().splitlines()


def get_version(initpath: str) -> str:
    """ Get from the init of the source code the version string

    Params:
        initpath (str): path to the init file of the python package relative to the setup file

    Returns:
        str: The version string in the form 0.0.1
    """

    path = os.path.join(os.path.dirname(__file__), initpath)

    with open(path, "r") as handle:
        for line in handle.read().splitlines():
            if line.startswith("__version__"):
                return line.split("=")[1].strip().strip("\"'")
        else:
            raise RuntimeError("Unable to find version string.")


setup(
    name="mltu",
    version=get_version("mltu/__init__.py"),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://pylessons.com/",
    author="PyLessons",
    author_email="pythonlessons0@gmail.com",
    install_requires=requirements,
    extras_require={
        "gpu": ["onnxruntime-gpu"],
    },
    python_requires=">=3",
    packages=find_packages(exclude=("*_test.py",)),
    include_package_data=True,
    project_urls={
        "Source": "https://github.com/pythonlessons/mltu/",
        "Tracker": "https://github.com/pythonlessons/mltu/issues",
    },
    description="Machine Learning Training Utilities (MLTU) for TensorFlow and PyTorch",
)
