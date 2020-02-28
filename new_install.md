# Installation notes

To install fortran compiler on Mac:

```bash
brew update
brew upgrade
brew info gcc
brew install gcc
brew cleanup
```

To create a conda environment to run the script in the `example/` directory:

```bash
conda create -n test_f2py_2020 python=3.7
conda install -c anaconda numpy
conda install -c conda-forge scipy
conda install -c conda-forge netcdf4
conda install -c conda-forge pytest
conda install -c conda-forge matplotlib=2.2.2
conda install -c bioconda ecmwfapi
conda install -c conda-forge jupyter
```