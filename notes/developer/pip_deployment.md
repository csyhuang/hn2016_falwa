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

```
 