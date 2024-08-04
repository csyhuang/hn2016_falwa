project('falwa')

# Based on:
# - https://numpy.org/doc/stable/f2py/buildtools/meson.html
# - https://meson-python.readthedocs.io/en/stable/how-to-guides/first-project.html
# - https://github.com/jameskermode/f90wrap/blob/master/f90wrap/meson.build

add_languages('c')
add_languages('fortran')

# Python configuration
py = import('python').find_installation(pure: false)

inc_root = include_directories('.')

subdir('src')
