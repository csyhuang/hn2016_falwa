

Object Oriented Interface
==========================

.. automodule:: oopinterface
   :members:


Terms in the LWA Column Budget
------------------------------

There are two sets of methods to compute the LWA column budget available in this class:

1. `Nakamura and Huang, Science (2018) <https://www.science.org/doi/10.1126/science.aat0721>`_ : eq (2), (3) with reference states solved with boundary conditions (S4).
    The corresponding methods of flux computation are:
        1. `interpolate_fields`
        2. `compute_reference_states`: matrix inversion is done using `SOR <https://github.com/csyhuang/hn2016_falwa/blob/master/notes/SOR_solver_for_NH18.pdf>`_
        3. `compute_lwa_and_barotropic_fluxes`


2. `Neal et al., GRL (2022) <https://doi.org/10.1029/2021GL097699>`_ : eq (S5) - (S7) with reference states solved with boundary conditions (S14) - (S16).
    The corresponding methods of flux computation are:
        1. `_interpolate_field_dirinv`
        2. `_compute_qref_fawa_and_bc`: since the latitudinal boundary is now away from the equator, solution is guaranteed such that `Direct Inversion <https://github.com/csyhuang/hn2016_falwa/blob/master/notes/Direct_solver_for_NHN22.pdf>`_ is implemented.
        3. `_compute_lwa_flux_dirinv`

With the LWA column budget as formulated by `Nakamura and Huang, Science (2018) <https://www.science.org/doi/10.1126/science.aat0721>`_, the output fields of :py:meth:`QGField.compute_lwa_and_barotropic_fluxes` correspond to terms as follows:

.. math::

   \frac{\partial}{\partial t} \boxed{ \langle A \rangle \cos(\phi) }_\mathtt{~lwa\_baro} ~ = ~
        & \boxed{ - \frac{1}{a \cos(\phi)} \frac{\partial}{\partial \lambda} \langle F_{\lambda} \rangle }_\mathtt{~convergence\_zonal\_advective\_flux} \\
        & \boxed{ + \frac{1}{a \cos(\phi)} \frac{\partial}{\partial \phi'} \langle u_e v_e \cos^2(\phi + \phi') \rangle }_\mathtt{~divergence\_eddy\_momentum\_flux} \\
        & \boxed{ + \frac{f \cos(\phi)}{H} \left( \frac{v_e \theta_e}{\partial \tilde\theta / \partial z} \right)_{z=0} }_\mathtt{~meridional\_heat\_flux} \\
        & + \langle \dot A \rangle \cos(\phi) \\

   \langle F_{\lambda} \rangle ~ = ~
        & \boxed{ \langle u_\mathrm{REF} A \cos(\phi) \rangle }_\mathtt{~adv\_flux\_f1} \\
        & \boxed{ - a \left\langle \int_0^{\Delta\phi} u_e q_e \cos(\phi + \phi') \mathrm{d}\phi' \right\rangle }_\mathtt{~adv\_flux\_f2} \\
        & \boxed{ + \frac{\cos(\phi)}{2} \left\langle v_e^2 - u_e^2 - \frac{R}{H} \frac{e^{-\kappa z / H} \theta_e^2}{\partial \tilde\theta / \partial z} \right\rangle }_\mathtt{~adv\_flux\_f3} \\


.. note::
    
    The `dirinv`-based routines used in `Neal et al., GRL (2022) <https://doi.org/10.1029/2021GL097699>`_ added in release 0.6.0 are encapsulated in implicit functions (which will be public in release 0.7.0).

    The output fields of :py:meth:`QGField._compute_lwa_flux_dirinv` correspond to the terms of the LWA budget in the following way:

    .. math::

       \frac{\partial}{\partial t} \boxed{ \langle A \rangle \cos(\phi) }_\mathtt{~astarbaro} ~ = ~
            & - \frac{1}{a \cos(\phi)} \frac{\partial}{\partial \lambda} \langle F_{\lambda} \rangle \\
            & - \frac{1}{a \cos(\phi)} \frac{\partial}{\partial \phi'} \boxed{ \langle F_{\phi'} \cos(\phi + \phi') \rangle }_{~\mathtt{~ep2baro} \mathrm{(north)}, \mathtt{ep3baro} \mathrm{(south)}} \\
            & \boxed{ + \frac{f \cos(\phi)}{H} \left( \frac{v_e \theta_e}{\partial \tilde\theta / \partial z} \right)_{z=0} }_\mathtt{~ep4} \\
            & + \langle \dot A \rangle \cos(\phi) \\

       \langle F_{\lambda} \rangle ~ = ~
            & \boxed{ \langle u_\mathrm{REF} A \cos(\phi) \rangle }_\mathtt{~ua1baro} \\
            & \boxed{ - a \left\langle \int_0^{\Delta\phi} u_e q_e \cos(\phi + \phi') \mathrm{d}\phi' \right\rangle }_\mathtt{~ua2baro} \\
            & \boxed{ + \frac{\cos(\phi)}{2} \left\langle v_e^2 - u_e^2 - \frac{R}{H} \frac{e^{-\kappa z / H} \theta_e^2}{\partial \tilde\theta / \partial z} \right\rangle }_\mathtt{~ep1baro} \\

       \langle F_{\phi'} \rangle ~ = ~
            & - \langle u_e v_e A \cos(\phi + \phi') \rangle

    At the mean time, if you want to use this set of routine, please refer to ``scripts/nhn_grl2022/sample_run_script.py`` for the usage.