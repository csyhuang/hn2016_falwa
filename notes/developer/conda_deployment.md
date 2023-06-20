# Conda package deployment

## Necessary files

```
├── hn2016_falwa
│   ├── recipe
│   |   ├── build.sh
│   |   ├── meta.yaml
│   |   ├── bld.bat (for windows)
```

## Changes in setup.py

On master branch, the arguments for `setup` in `setup.py`:

```python
from setuptools import find_packages

setup(
    ...
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'xarray'],
    setup_requires=['pytest-runner'],
    ...
)
```
is changed to (the version numbers below come from [MDTF base environment](https://github.com/NOAA-GFDL/MDTF-diagnostics/blob/main/src/conda/env_base.yml))
```python
setup(
    ...
    python_requires='>=3',
    packages=['hn2016_falwa', 'tests', 'hn2016_falwa.legacy'],
    setup_requires=['numpy==1.22.3'],
    install_requires=['numpy==1.22.3', 'scipy==1.9', 'xarray==2023.2.0'],
    ...
)
```
Not sure if all changes are necessary though. I keep a copy of setup script that works in this directory: `setup_deployment.py` for future reference.

## Conda packaging

These procedures work on Linux-x64 so far. (No successful attempt on OSX-64.)

To prepare for build environment
```bash
conda install conda-build  # for building the package for deployment onto anaconda
conda install anaconda-client  # for uploading the tar.bz2 package onto anaconda channel
```

To trigger the build, execute in `hn2016_falwa/`:
```bash
conda build recipe/
```

To convert the package to format suitable for all platforms:
```bash
conda convert -f --platform all $HOME/miniconda3/envs/test_env/conda-bld/linux-64/hn2016_falwa-0.7.0-py310_0.tar.bz2 -o outputdir/
```
You can then find all the distributions in the directory `outputdir/`.

To upload the packages, login with `anaconda login`. Then, do: 

```bash
anaconda upload osx-64/hn2016_falwa-0.7.0-py310_0.tar.bz2
```
