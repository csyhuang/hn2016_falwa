Xarray Interface
================

A wrapper to manage multiple :py:class:`.oopinterface.QGField` objects, with
`xarray <https://xarray.dev/>`_-based in- and output.

.. note::

    The xarray interface is neither memory- nor dask-optimized. For large
    datasets it might be necessary to manually split the data and process them
    chunk-by-chunk to avoid running out of main memory. See
    `Issue #50 <https://github.com/csyhuang/hn2016_falwa/issues/50>`_ on GitHub
    where progress on memory optimization is tracked.


.. automodule:: xarrayinterface
   :members:

