

Preprocessing
==============

Since `QGField` requires all grid points of the input fields to have finite values, to process model output with grid points below topography masked as `NaN`, users have to fill in reasonable values for those grid points.

A pragmatic implementation was published in `Lubis et al (2018) <https://journals.ametsoc.org/view/journals/clim/31/10/jcli-d-17-0382.1.xml>`_, in which the grids are filled level-by-level by solving Poisson equation, such that no extremum values are filled in. (In contrast, extrapolation over vertical direction would introduce unrealistic extreme values.) Such gridfill procedure is implemented by the python package `gridfill <https://github.com/ajdawson/gridfill>`_ published by Andrew Dawson. After computing the finite-amplitude wave diagnostics, regions with gridfilled values are masked out.

