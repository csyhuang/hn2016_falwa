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

 