===================
MoCap Object Frames
===================

Grasp trajectories should be expressed in an object-centric coordinate system.
Hence, we have to define the frame of each grasped object. We do this in
the module ``hand_embodiment.mocap_objects``.

Each MoCap object has a set of markers associated with it. Placement on the
object is indicated in a docstring of the corresponding class (in a top view).
For each object we define default marker positions in the object coordinate
system and a function to reconstruct the pose of the object from actual marker
positions. The pose of the object is the pose of the marker frame in the MoCap
world frame.

Furthermore, for visualization we need meshes and their relation to the
marker frame. These are defined in ``hand_embodiment.vis_utils``. Each
object visualization class contains the transformation ``markers2mesh`` as
an attribute. Furthermore, it loads the mesh in its constructor along with
the correct color.

Here are the currently supported objects. In each object you see the mesh,
the markers used to track the object's pose and the frame defined based on
the marker positions.

Available Objects
-----------------

Electronic Components
^^^^^^^^^^^^^^^^^^^^^

.. image:: _static/object_frames/electronic_object.png

.. image:: _static/object_frames/electronic_target.png

OSAI Case
^^^^^^^^^

.. image:: _static/object_frames/osai_case.png

Small OSAI Case
^^^^^^^^^^^^^^^

.. image:: _static/object_frames/osai_case_small.png

Open Passport
^^^^^^^^^^^^^

.. image:: _static/object_frames/passport_open.png

Closed Passport
^^^^^^^^^^^^^^^

.. image:: _static/object_frames/passport_closed.png

Passport Box
^^^^^^^^^^^^

.. image:: _static/object_frames/passport_box.png

Insole
^^^^^^

.. image:: _static/object_frames/insole.png

Small Pillow
^^^^^^^^^^^^

.. image:: _static/object_frames/pillow_small.png

Big Pillow
^^^^^^^^^^

.. image:: _static/object_frames/pillow_big.png
