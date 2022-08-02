===============================
Configuration for a New Subject
===============================

Each (human) subject has a different hand and we have to take this into account
in the configuration of the record mapping, that is, we have to adapt the shape
of the MANO model. For this purpose there is the script ``bin/gui_mano_shape.py``
with which you can load a Qualisys tsv file, visualize the hand markers,
and visualize and modify the shape and pose of the MANO mesh with respect to
the hand marker's frame at the back of the hand. You can create new
configuration files or modify an existing one. Make sure to save the edited
configuration with the button in the menu (left top). The MANO configuration
will later on be used when we recover the state of the hand from the motion
capture data.

Parameter Configuration GUI
---------------------------

.. image:: _static/mano_shape.png
