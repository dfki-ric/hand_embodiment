============
Installation
============

Install Hand Embodiment
-----------------------

This software is implemented in Python. If you do not have a working Python
environment so far, we recommend to install it with
`Anaconda <https://www.anaconda.com/>`_. Then you can follow these
instructions:

**Clone repository:**

.. code-block:: bash

    git clone https://github.com/dfki-ric/hand_embodiment.git

**Prepare MANO model:**

1. Download MANO model from `here <https://mano.is.tue.mpg.de/>`_ and unzip it
   to ``hand_embodiment/hand_embodiment/model/mano``.
2. Go to ``hand_embodiment/hand_embodiment/model/mano``.

.. code-block:: bash

    cd hand_embodiment/hand_embodiment/model/mano

3. For this preparation step, please install following dependencies.

.. code-block::

    pip install numpy scipy chumpy

4. Run ``prepare_mano.py``, you will get the converted MANO model that
   is compatible with this project.

.. code-block:: bash

    python prepare_mano.py  # convert model to JSON

5. Go back to the main folder.

.. code-block:: bash

    cd ../../../

**Install hand_embodiment from main directory:**

.. code-block:: bash

    pip install -e .

The Python version used to produce the results in the paper was 3.8. Exact
versions of the dependencies are listed in
requirements.txt (``pip install -r requirements.txt``).
Alternatively, you can create a conda environment from
environment.yml:

.. code-block:: bash

    conda env create -f environment.yml
    conda activate hand_embodiment
    pip install -e .
    pytest test/  # verify installation

Data
----

The dataset is available at Zenodo:

.. image:: https://zenodo.org/badge/DOI/10.5281/zenodo.7130208.svg
   :target: https://doi.org/10.5281/zenodo.7130208

We recommend to use `zenodo_get <https://gitlab.com/dvolgyes/zenodo_get>`_ to
download the data:

.. code-block:: bash

    pip install zenodo-get
    mkdir -p some/folder/in/which/the/data/is/located
    cd some/folder/in/which/the/data/is/located
    zenodo_get 7130208
