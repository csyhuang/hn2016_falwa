# What I did on Mac

I did the same thing as in `conda_deployment.md` up to this point:

To trigger the build, execute in `hn2016_falwa/` `conda build recipe/`.

# Errors

```
ld: unsupported tapi file type '!tapi-tbd' in YAML file '/Library/Developer/CommandLineTools/SDKs/MacOSX11.sdk/usr/lib/libSystem.tbd' for architecture x86_64
```

## SDK version

To find out the SDK version: 
```
$ xcodebuild -sdk -version
```
This outputs `xcodebuild-sdk-version.log`. The default is MacOSX12.1.sdk - macOS 12.1 (macosx12.1).

## Problem solving

- https://stackoverflow.com/questions/69236331/conda-macos-big-sur-ld-unsupported-tapi-file-type-tapi-tbd-in-yaml-file

Seemed to be a solution:
- https://www.appsloveworld.com/cplus/100/303/conda-macos-big-sur-ld-unsupported-tapi-file-type-tapi-tbd-in-yaml-file

Changes done before:
- https://www.appsloveworld.com/cplus/100/303/conda-macos-big-sur-ld-unsupported-tapi-file-type-tapi-tbd-in-yaml-file

# Trying other solutions

Install gfortran in `intel_cython`:

```bash
conda install -c conda-forge gfortran
```

# Try conda packaging with GitHub actions

## Trial 1
Try this GitHub action: https://github.com/marketplace/actions/build-and-publish-conda-packages-to-anaconda-org


## Added this to conda_build_config.yaml

Modify `/.conda/conda_build_config.yaml` accordingly:

> https://stackoverflow.com/questions/53196129/conda-forge-recipe-for-python-package-with-a-fortran-extension-not-working-on-ap