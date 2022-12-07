

Object Oriented Interface
==========================

.. automodule:: oopinterface
   :members:


Terms in the LWA Column Budget
------------------------------

With the LWA column budget as formulated by `Neal et al. <https://doi.org/10.1029/2021GL097699>`_ (2022, supporting information, equations 5-7), the output fields of :py:meth:`QGField.compute_lwa_and_barotropic_fluxes` correspond to terms as follows:

.. math::

   \frac{\partial}{\partial t} \boxed{ \langle A \rangle \cos(\phi) }_\mathtt{~lwa\_baro} ~ = ~
        & \boxed{ - \frac{1}{a \cos(\phi)} \frac{\partial}{\partial \lambda} \langle F_{\lambda} \rangle }_\mathtt{~convergence\_zonal\_advective\_flux} \\
        & \boxed{ - \frac{1}{a \cos(\phi)} \frac{\partial}{\partial \phi'} \langle F_{\phi'} \cos(\phi + \phi') \rangle }_\mathtt{~divergence\_eddy\_momentum\_flux} \\
        & \boxed{ + \frac{f \cos(\phi)}{H} \left( \frac{v_e \theta_e}{\partial \tilde\theta / \partial z} \right)_{z=0} }_\mathtt{~meridional\_heat\_flux} \\
        & + \langle \dot A \rangle \cos(\phi) \\

   \langle F_{\lambda} \rangle ~ = ~
        & \boxed{ \langle u_\mathrm{REF} A \cos(\phi) \rangle }_\mathtt{~adv\_flux\_f1} \\
        & \boxed{ - a \left\langle \int_0^{\Delta\phi} u_e q_e \cos(\phi + \phi') \mathrm{d}\phi' \right\rangle }_\mathtt{~adv\_flux\_f2} \\
        & \boxed{ + \frac{\cos(\phi)}{2} \left\langle v_e^2 - u_e^2 - \frac{R}{H} \frac{e^{-\kappa z / H} \theta_e^2}{\partial \tilde\theta / \partial z} \right\rangle }_\mathtt{~adv\_flux\_f3} \\

   \langle F_{\phi'} \rangle ~ = ~
        & - \langle u_e v_e A \cos(\phi + \phi') \rangle


.. note::
    
    The `dirinv`-based routines added in version 0.6 as an alternative to the `SOR`-based routines are still considered experimental.
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

