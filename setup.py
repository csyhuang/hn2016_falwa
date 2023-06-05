# https://setuptools.pypa.io/en/latest/userguide/ext_modules.html

from numpy.distutils.core import setup, Extension


ext1 = Extension(name='hn2016_falwa.interpolate_fields',
                 sources=['hn2016_falwa/f90_modules/interpolate_fields.f90'],
                 f2py_options=['--quiet'])

ext2 = Extension(name='hn2016_falwa.compute_reference_states',
                 sources=['hn2016_falwa/f90_modules/compute_reference_states.f90'],
                 f2py_options=['--quiet'])

ext3 = Extension(name='hn2016_falwa.compute_lwa_and_barotropic_fluxes',
                 sources=['hn2016_falwa/f90_modules/compute_lwa_and_barotropic_fluxes.f90'],
                 f2py_options=['--quiet'])

# *** Extensions 4-9 are used by the direct inversion algorithm ***
ext4 = Extension(name='hn2016_falwa.interpolate_fields_direct_inv',
                 sources=['hn2016_falwa/f90_modules/interpolate_fields_dirinv.f90'],
                 f2py_options=['--quiet'])

ext5 = Extension(name='hn2016_falwa.compute_qref_and_fawa_first',
                 sources=['hn2016_falwa/f90_modules/compute_qref_and_fawa_first.f90'],
                 f2py_options=['--quiet'])

ext6 = Extension(name='hn2016_falwa.matrix_b4_inversion',
                 sources=['hn2016_falwa/f90_modules/matrix_b4_inversion.f90'],
                 f2py_options=['--quiet'])

ext7 = Extension(name='hn2016_falwa.matrix_after_inversion',
                 sources=['hn2016_falwa/f90_modules/matrix_after_inversion.f90'],
                 f2py_options=['--quiet'])

ext8 = Extension(name='hn2016_falwa.upward_sweep',
                 sources=['hn2016_falwa/f90_modules/upward_sweep.f90'],
                 f2py_options=['--quiet'])

ext9 = Extension(name='hn2016_falwa.compute_flux_dirinv',
                 sources=['hn2016_falwa/f90_modules/compute_flux_dirinv.f90'],
                 f2py_options=['--quiet'])

setup(
    ext_modules=[ext1, ext2, ext3, ext4, ext5, ext6, ext7, ext8, ext9],
)

