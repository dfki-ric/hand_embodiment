==========
Unit Tests
==========

There are unit tests for this library. You need to install

.. code-block:: bash

    pip install -e .[test]

to run them. Then you can run the tests from the main folder with

.. code-block:: bash

    pytest test/

You can generate a coverage report with

.. code-block:: bash

    pytest --cov-report html --cov=hand_embodiment test/
