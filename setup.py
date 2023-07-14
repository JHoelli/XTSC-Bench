import io
import os
import subprocess
import sys
#TODO Quantus and TSSynth and Plotly
import setuptools
#subprocess.check_call([sys.executable, "-m", "pip", "install", "Cython"])
#try:
#    from numpy import get_include
#except ImportError:
#    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy ==1.21.6"])
#    from numpy import get_include


# Package meta-data.
NAME = "TSInterpret"
DESCRIPTION = "todo"
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
URL = "https://ipe-wim-gitlab.fzi.de/hoellig/benchmarkingtimeseries"
EMAIL = "hoellig@fzi.de"
AUTHOR = "Jacqueline Hoellig"
REQUIRES_PYTHON = ">=3.8.0"

# Package requirements.
base_packages = [
    "tsinterpret==0.3.1"

]

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = "\n" + f.read()

# Load the package's __version__.py module as a dictionary.
about = {}
with open(os.path.join(here, NAME, "__version__.py")) as f:
    exec(f.read(), about)

# Where the magic happens:
setuptools.setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type=LONG_DESCRIPTION_CONTENT_TYPE,
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    packages=setuptools.find_packages(exclude=("tests",)),
    install_requires=base_packages,
    extras_require={
        ":python_version == '3.8'": ["dataclasses"],
    },
    include_package_data=True,
    license="BSD-3",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    ext_modules=[]
)