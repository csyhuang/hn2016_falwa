# Deployment onto PyPI

To compile both the source distribution and `.whl` file:

```bash
python3 setup.py sdist bdist_wheel
```

To upload the package onto TestPyPI to test deployment: 
```
python3 -m twine upload --repository testpypi dist/*
```

Deploy the package onto PyPI for real: 
```
python3 -m twine upload dist/*
```
# Update of deployment after using pyproject.toml

```
pip install build
python -m build . --sdist
python3 -m pip install --upgrade twine
python3 -m twine upload --repository pypi dist/*
```



```error
clare@otc:~/isolated_test/hn2016_falwa$ python3 -m pip install --index-url https://test.pypi.org/simple/ --no-deps falwa
Looking in indexes: https://test.pypi.org/simple/
Collecting falwa
  Downloading https://test-files.pythonhosted.org/packages/b1/0c/00cd0cd844c7d5c2faed03b3a1051de15e42afabe128783c1b9eb8239651/falwa-1.4.0a0.tar.gz (46.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 46.5/46.5 MB 625.8 kB/s eta 0:00:00
  Installing build dependencies ... error
  error: subprocess-exited-with-error
  
  × pip subprocess to install build dependencies did not run successfully.
  │ exit code: 1
  ╰─> [13 lines of output]
      Looking in indexes: https://test.pypi.org/simple/
      Collecting numpy
        Downloading https://test-files.pythonhosted.org/packages/d5/80/b947c574d9732e39db59203f9aa35cb4d9a5dd8a0ea2328acb89cf10d6e3/numpy-1.9.3.zip (4.5 MB)
           ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.5/4.5 MB 714.2 kB/s eta 0:00:00
        Preparing metadata (setup.py): started
        Preparing metadata (setup.py): finished with status 'done'
      Collecting meson-python
        Downloading https://test-files.pythonhosted.org/packages/c9/b6/9665154ee9926317a248e2b171ea21ac2b77788adea404566eec29b84f3b/meson_python-0.13.0-py3-none-any.whl.metadata (4.1 kB)
      Collecting meson>=0.63.3 (from meson-python)
        Downloading https://test-files.pythonhosted.org/packages/ae/c6/e3c2fa2fc539ca99679f02b05700b56c76ffb9338c1dd62f1c64391828ba/meson-1.1.991-py3-none-any.whl.metadata (1.8 kB)
      INFO: pip is looking at multiple versions of meson-python to determine which version is compatible with other requirements. This could take a while.
      ERROR: Could not find a version that satisfies the requirement pyproject-metadata>=0.7.1 (from meson-python) (from versions: none)
      ERROR: No matching distribution found for pyproject-metadata>=0.7.1
      [end of output]
  
  note: This error originates from a subprocess, and is likely not a problem with pip.
error: subprocess-exited-with-error

× pip subprocess to install build dependencies did not run successfully.
│ exit code: 1
╰─> See above for output.

note: This error originates from a subprocess, and is likely not a problem with pip.
``` 

Successfully installed from `.tar.gz`:

```bash
conda install anaconda::pip
python3 -m pip install falwa-1.4.0a0.tar.gz 
pytest tests/
```