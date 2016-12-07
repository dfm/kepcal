Installation
------------

Start by installing the dependencies:

1. numpy
2. pandas (+ PyTables)
3. matplotlib
4. `astropy <http://www.astropy.org/>`_
5. `fitsio <https://github.com/esheldon/fitsio>`_
6. `tqdm <https://github.com/noamraph/tqdm>`_


Get the data
------------

To download the Kepler full frame images, run::

    wget -r -nH -nd -np -R index.html -e robots=off https://archive.stsci.edu/pub/kepler/ffi/

in the ``$KEPCAL_DATA/ffi`` directory. This will take a long time and use
about 44Gb of space. Then, execute::

    scripts/kepcal-index-ffis $KEPCAL_DATA

to build an index of the FFI files.

You should also download and extract the `Kepler Input Catalog
<http://archive.stsci.edu/pub/kepler/catalogs/kic.txt.gz>`_ into the
``$KEPCAL_DATA/kic`` directory. Then run::

    scripts/kepcal-format-kic $KEPCAL_DATA

to save the KIC into a binary format that will be way faster to read.


