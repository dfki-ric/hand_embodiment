====================
Motion Capture Setup
====================

Glove
-----

Human fingers (index, middle, ring, and little finger) have three joints.
However, isolated joint movements are not always possible. In particular,
the last two finger joints often move together. While it is very difficult
to isolate movements of the last finger joints (distal interphalangeal, between
intermediate phalanges and distal phalanges), it is possible to isolate
movements of the proximal interphalangeal joints (between proximal phalanges
and intermediate phalanges) and the metacarpophalangeal joints (between
metacarpal bones and proximal phalanges).

.. image:: _static/finger_tendons.png

Tendons of finger flexors and extensor. Yellow: tendons of flexor digitorum
profundus attach to distal phalanges. Red: tendons of extensor digitorum also
attach to the distal phalanges. Orange: tendons of the flexor digitorum
superficialis attach to the intermediate phalanges.

We conclude that it is enough to attach two markers to the fingers to track
their state of flexion or extension. Furthermore, with these two markers we
also can measure their state of abduction or adduction.

We used the following labelled markers on the glove to record the datasets used
in the paper:

* Hand pose: ``hand_top``, ``hand_left``, ``hand_right``
* Per finger (thumb, index, middle, ring, little): ``[finger]_middle``,
  ``[finger]_tip``

The following shows the configuration of our glove for the right hand:

.. image:: _static/glove.png

The figure is based on
`this work <https://commons.wikimedia.org/wiki/File:Scheme_human_hand_bones-en.svg>`_
of Mariana Ruiz Villarreal (LadyofHats); retouches by Nyks.

Tracked Objects
---------------

In addition, we attached motion capture markers to objects to track their poses
and to transfer object manipulation trajectories into an object-relative
coordinate system. The following image shows these objects.

.. image:: _static/objects.png

The marker configuration and the definition of frames based on markers is
implemented and documented in
`hand_embodiment/mocap_objects <https://github.com/dfki-ric/hand_embodiment/blob/main/hand_embodiment/mocap_objects.py>`_.
We assume that the z-axis points up for all objects with only two markers.
Meshes for visualization are available
`here <https://github.com/dfki-ric/hand_embodiment/tree/main/hand_embodiment/model/objects>`_
and transformations between marker frames and the mesh frames can be found in
`hand_embodiment/vis_utils <https://github.com/dfki-ric/hand_embodiment/blob/main/hand_embodiment/vis_utils.py>`_.
