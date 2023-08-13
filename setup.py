import io
import os
import subprocess
import sys

import setuptools
#subprocess.check_call([sys.executable, "-m", "pip", "install", "Cython"])
#try:
#    from numpy import get_include
#except ImportError:
#    subprocess.check_call([sys.executable, "-m", "pip", "install", "numpy ==1.21.6"])
#    from numpy import get_include


# Package meta-data.
NAME = "XTSCBench"
DESCRIPTION = ""
LONG_DESCRIPTION_CONTENT_TYPE = "text/markdown"
URL = "https://"
EMAIL = "."
AUTHOR = "."
REQUIRES_PYTHON = ">3.9.0"

# Package requirements.
base_packages = [
    "timesynth @ https://github.com/TimeSynth/TimeSynth/archive/refs/heads/master.zip ",
    "ipykernel==6.16.0",
    "ipython==8.5.0",
    "ipython-genutils==0.2.0",
    "kaleido==0.2.1",
    "matplotlib==3.7.2",
    "numba==0.57.1",
    "numpy==1.22.1",
    "pandas==1.3.5",
    "plotly==5.15.0",
    #"scikit-image==0.21.0",
    #"scikit-learn==1.2.0",
    #"scikit-optimize==0.9.0",
    "scipy==1.11.1",
    "seaborn==0.12.2",
    "sklearn==0.0",
    "torch",
    #"TSInterpret>0.3.0",
    "quantus==0.4.1",
    "cachetools",
    "tslearn"
    
]

here = os.path.abspath(os.path.dirname(__file__))

# Import the README and use it as the long-description.
with io.open(os.path.join(here, "ReadMe.md"), encoding="utf-8") as f:
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
        ":python_version == '3.9'": ["dataclasses"],
    },
    include_package_data=True,
    license="BSD-3",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
        "Programming Language :: Python :: Implementation :: PyPy",
    ],
    ext_modules=[]
)
